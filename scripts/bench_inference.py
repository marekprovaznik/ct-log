import argparse
import json
from pathlib import Path
import platform
import re
import subprocess
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
from src.configs.kwp_training_config import KwpTrainingConfig
import torch
import torch.nn.functional as F

try:
    import torch_tensorrt
except ImportError:
    torch_tensorrt = None

NATIVE = 778
PAGE_RE = re.compile(r"page_(\d+)")
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
VARIANTS: Dict[str, Dict] = {
    "fp32": {"dtype": "fp32", "tf32": False, "channels_last": False, "compile": None},
    "tf32": {"dtype": "fp32", "tf32": True, "channels_last": False, "compile": None},
    "bf16_autocast": {"dtype": "bf16_autocast", "tf32": False, "channels_last": False, "compile": None},
    "fp16_autocast": {"dtype": "fp16_autocast", "tf32": False, "channels_last": False, "compile": None},
    "fp16": {"dtype": "fp16", "tf32": False, "channels_last": False, "compile": None},
    "fp16_cl": {"dtype": "fp16", "tf32": False, "channels_last": True, "compile": None},
    "bf16_cl": {"dtype": "bf16", "tf32": False, "channels_last": True, "compile": None},
    "fp16_cl_compile": {"dtype": "fp16", "tf32": False, "channels_last": True, "compile": "max-autotune-no-cudagraphs"},
    "fp16_cl_compile_cg": {"dtype": "fp16", "tf32": False, "channels_last": True, "compile": "max-autotune"},
    "trt_fp16": {"dtype": "fp16", "tf32": False, "channels_last": False, "compile": "tensorrt"},
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M5 inference speed: model only and end to end, per variant and batch size."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--img_dir", type=Path, required=True, help="Log directory with page_*.tiff (log 10).")
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--batch_sizes", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--accuracy_batch", type=int, default=4)
    parser.add_argument("--host", required=True, help="Machine label recorded in the output (e.g. precision, euler).")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    torch._dynamo.config.cache_size_limit = 64
    config = KwpTrainingConfig.from_yaml(args.config)
    stacks = load_log(args.img_dir, config.window)
    result: Dict = {"env": environment(args, config, stacks.shape[0]), "runs": [], "accuracy": {}}
    reference: Optional[Tuple[np.ndarray, np.ndarray]] = None
    for name in args.variants:
        spec = VARIANTS[name]
        set_tf32(spec["tf32"])
        net = build_net(config, args.checkpoint, spec, device)
        pipeline = Pipeline(net, spec, tuple(config.resolution), config.num_classes, device)
        classes, pith = pipeline.predict_log(stacks, args.accuracy_batch)
        if reference is None:
            reference = (classes, pith)
        result["accuracy"][name] = deviation(classes, pith, *reference)
        print("%s accuracy vs %s: %s" % (name, args.variants[0], json.dumps(result["accuracy"][name])), flush=True)
        for batch_size in args.batch_sizes:
            for mode in ("model", "e2e"):
                run = {"variant": name, **spec, "batch_size": batch_size, "mode": mode}
                try:
                    run.update(benchmark(pipeline, stacks, batch_size, mode, args.warmup, args.iters, args.repeats))
                except torch.cuda.OutOfMemoryError:
                    run["error"] = "oom"
                    torch.cuda.empty_cache()
                result["runs"].append(run)
                print(summary(run), flush=True)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(result, indent=1))
        del pipeline, net
        torch.cuda.empty_cache()
    print("wrote %s" % args.out)


class Pipeline:
    def __init__(
        self, net: torch.nn.Module, spec: Dict, resolution: Tuple[int, int], num_classes: int, device: torch.device
    ):
        self.net = net
        self.spec = spec
        self.resolution = resolution
        self.num_classes = num_classes
        self.device = device
        self.input_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(spec["dtype"], torch.float32)
        self.autocast_dtype = {"fp16_autocast": torch.float16, "bf16_autocast": torch.bfloat16}.get(spec["dtype"])
        self.memory_format = torch.channels_last if spec["channels_last"] else torch.contiguous_format
        self.mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    def preprocess(self, stacks_u8: torch.Tensor) -> torch.Tensor:
        images = stacks_u8.to(self.device, non_blocking=True).float() / 255.0
        images = F.interpolate(images, size=self.resolution, mode="bilinear", align_corners=False)
        images = (images - self.mean) / self.std
        return images.to(dtype=self.input_dtype, memory_format=self.memory_format)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        with torch.autocast(
            "cuda", dtype=self.autocast_dtype or torch.float16, enabled=self.autocast_dtype is not None
        ):
            return self.net(images)

    def postprocess(self, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = logits.float()
        probs = logits[:, : self.num_classes].softmax(dim=1)
        probs = F.interpolate(probs, size=(NATIVE, NATIVE), mode="bilinear", align_corners=False)
        classes = probs.argmax(dim=1).to(torch.uint8)
        pith = soft_argmax(logits[:, self.num_classes]) * NATIVE
        return classes.cpu(), pith.cpu()

    def end_to_end(self, stacks_u8: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.postprocess(self.forward(self.preprocess(stacks_u8)))

    @torch.inference_mode()
    def predict_log(self, stacks: torch.Tensor, batch_size: int) -> Tuple[np.ndarray, np.ndarray]:
        classes, piths = [], []
        for start in range(0, stacks.shape[0], batch_size):
            batch = stacks[start : start + batch_size]
            if batch.shape[0] < batch_size:
                batch = torch.cat([batch, batch[-1:].expand(batch_size - batch.shape[0], -1, -1, -1)])
            cls, pith = self.end_to_end(batch)
            classes.append(cls.numpy())
            piths.append(pith.numpy())
        count = stacks.shape[0]
        return np.concatenate(classes)[:count], np.concatenate(piths)[:count]


def load_log(img_dir: Path, window: int) -> torch.Tensor:
    pages = sorted((int(PAGE_RE.search(p.stem).group(1)), p) for p in img_dir.glob("*.tiff") if PAGE_RE.search(p.stem))
    if not pages:
        msg = "No page_*.tiff in %s" % img_dir
        raise ValueError(msg)
    slices = np.stack([np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for _, path in pages])
    count = slices.shape[0]
    offsets = range(-window, window + 1) if window > 0 else (0, 0, 0)
    index = np.array([[min(max(i + o, 0), count - 1) for o in offsets] for i in range(count)])
    return torch.from_numpy(slices[index]).pin_memory()


def build_net(config: KwpTrainingConfig, checkpoint: Path, spec: Dict, device: torch.device) -> torch.nn.Module:
    extra = {"decoder_channels": list(config.unet_decoder_channels)} if config.unet_decoder_channels else {}
    net = smp.create_model(
        config.unet_arch,
        encoder_name=config.unet_encoder,
        encoder_weights=None,
        in_channels=3 if config.window <= 1 else 2 * config.window + 1,
        classes=config.num_classes + int(config.pith_regression),
        **extra,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["seg_head"]
    net.load_state_dict({key.removeprefix("net."): value for key, value in state.items()})
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(spec["dtype"], torch.float32)
    memory_format = torch.channels_last if spec["channels_last"] else torch.contiguous_format
    net = net.to(device=device, dtype=dtype, memory_format=memory_format).eval()
    if spec["compile"] == "tensorrt":
        return TensorRTNet(net)
    if spec["compile"]:
        net = torch.compile(net, mode=spec["compile"], dynamic=False)
    return net


class TensorRTNet(torch.nn.Module):
    def __init__(self, net: torch.nn.Module):
        super().__init__()
        if torch_tensorrt is None:
            msg = "torch_tensorrt is not installed"
            raise ValueError(msg)
        self.net = net
        self.engines: Dict[Tuple[int, ...], torch.nn.Module] = {}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        shape = tuple(images.shape)
        if shape not in self.engines:
            self.engines[shape] = torch_tensorrt.compile(
                self.net,
                ir="dynamo",
                inputs=[torch_tensorrt.Input(shape, dtype=images.dtype)],
                enabled_precisions={images.dtype},
                optimization_level=5,
                min_block_size=1,
            )
        return self.engines[shape](images)


@torch.inference_mode()
def benchmark(
    pipeline: Pipeline, stacks: torch.Tensor, batch_size: int, mode: str, warmup: int, iters: int, repeats: int
) -> Dict:
    count = stacks.shape[0]
    batches = [stacks[(np.arange(batch_size) + start) % count] for start in range(0, count, batch_size)]
    batches = [batch.pin_memory() for batch in batches]
    inputs = [pipeline.preprocess(batch) for batch in batches[:4]] if mode == "model" else []

    def step(i: int) -> None:
        if mode == "model":
            pipeline.forward(inputs[i % len(inputs)])
        else:
            pipeline.end_to_end(batches[i % len(batches)])

    compile_start = time.perf_counter()
    for i in range(warmup):
        step(i)
    torch.cuda.synchronize()
    warmup_s = time.perf_counter() - compile_start
    torch.cuda.reset_peak_memory_stats()
    gpu_before = gpu_state()
    latencies: List[float] = []
    throughputs: List[float] = []
    for _ in range(repeats):
        for i in range(iters):
            start = time.perf_counter()
            step(i)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
        start = time.perf_counter()
        for i in range(iters):
            step(i)
        torch.cuda.synchronize()
        throughputs.append(batch_size * iters / (time.perf_counter() - start))
    latency_ms = np.array(latencies) * 1000.0
    per_repeat = latency_ms.reshape(repeats, iters).mean(axis=1)
    return {
        "warmup_s": round(warmup_s, 2),
        "latency_ms_mean": round(float(latency_ms.mean()), 3),
        "latency_ms_p50": round(float(np.percentile(latency_ms, 50)), 3),
        "latency_ms_p95": round(float(np.percentile(latency_ms, 95)), 3),
        "fps_latency": round(batch_size * 1000.0 / float(latency_ms.mean()), 1),
        "fps_throughput": round(float(np.median(throughputs)), 1),
        "fps_throughput_repeats": [round(t, 1) for t in throughputs],
        "fps_latency_repeats": [round(batch_size * 1000.0 / float(m), 1) for m in per_repeat],
        "peak_mem_mb": round(torch.cuda.max_memory_allocated() / 2**20),
        "gpu_before": gpu_before,
        "gpu_after": gpu_state(),
    }


def deviation(classes: np.ndarray, pith: np.ndarray, ref_classes: np.ndarray, ref_pith: np.ndarray) -> Dict:
    knot, ref_knot = classes == 2, ref_classes == 2
    union = int(np.logical_or(knot, ref_knot).sum())
    pith_err = np.linalg.norm(pith - ref_pith, axis=1)
    return {
        "knot_pixel_iou_vs_ref": round(float(np.logical_and(knot, ref_knot).sum() / union) if union else 1.0, 6),
        "knot_pixels_differing": int(np.logical_xor(knot, ref_knot).sum()),
        "class_pixels_differing": int((classes != ref_classes).sum()),
        "pith_px_diff_mean": round(float(pith_err.mean()), 4),
        "pith_px_diff_max": round(float(pith_err.max()), 4),
        "pith_rounded_points_differing": int((np.rint(pith) != np.rint(ref_pith)).any(axis=1).sum()),
        "slices": int(classes.shape[0]),
    }


def soft_argmax(logits: torch.Tensor) -> torch.Tensor:
    batch_size, height, width = logits.shape
    probs = logits.float().flatten(1).softmax(dim=1).view(batch_size, height, width)
    xs = (torch.arange(width, device=probs.device, dtype=probs.dtype) + 0.5) / width
    ys = (torch.arange(height, device=probs.device, dtype=probs.dtype) + 0.5) / height
    return torch.stack([(probs.sum(dim=1) * xs).sum(dim=1), (probs.sum(dim=2) * ys).sum(dim=1)], dim=1)


def set_tf32(enabled: bool) -> None:
    torch.backends.cudnn.allow_tf32 = enabled
    torch.backends.cuda.matmul.allow_tf32 = enabled


def gpu_state() -> Dict:
    query = "memory.used,memory.total,utilization.gpu,temperature.gpu,clocks.sm,power.draw"
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=%s" % query, "--format=csv,noheader,nounits"], capture_output=True, text=True
    )
    values = [v.strip() for v in out.stdout.strip().splitlines()[0].split(",")] if out.returncode == 0 else []
    return dict(zip(query.split(","), values))


def environment(args: argparse.Namespace, config: KwpTrainingConfig, slices: int) -> Dict:
    query = "name,driver_version,power.limit,memory.total"
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=%s" % query, "--format=csv,noheader"], capture_output=True, text=True
    )
    return {
        "host": args.host,
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": out.stdout.strip(),
        "cpu": platform.processor() or platform.machine(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "smp": smp.__version__,
        "checkpoint": str(args.checkpoint),
        "config": str(args.config),
        "native_size": NATIVE,
        "model_resolution": list(config.resolution),
        "input": "uint8 3-slice stack [z-1, z, z+1] in pinned host memory",
        "e2e": "H2D, /255, bilinear 778->800, ImageNet normalisation, U-Net, softmax, bilinear 800->778, argmax, pith soft-argmax, D2H (TIFF decoding excluded)",
        "model_only": "U-Net forward on a preprocessed device tensor",
        "warmup": args.warmup,
        "iters": args.iters,
        "repeats": args.repeats,
        "accuracy_slices": slices,
        "accuracy_reference": args.variants[0],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def summary(run: Dict) -> str:
    if "error" in run:
        return "%-20s b=%-3d %-5s %s" % (run["variant"], run["batch_size"], run["mode"], run["error"])
    return "%-20s b=%-3d %-5s fps=%7.1f thr=%7.1f p50=%7.2fms p95=%7.2fms mem=%dMB repeats=%s" % (
        run["variant"],
        run["batch_size"],
        run["mode"],
        run["fps_latency"],
        run["fps_throughput"],
        run["latency_ms_p50"],
        run["latency_ms_p95"],
        run["peak_mem_mb"],
        run["fps_throughput_repeats"],
    )


if __name__ == "__main__":
    main()
