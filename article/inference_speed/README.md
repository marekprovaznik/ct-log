# M5 inference speed

M5 (smp U-Net, EfficientNetV2-S encoder, decoder 512-256-128-64-32; checkpoint `seg_head_epoch_47.pth`,
sha256 `228ed1059172…`) segments CT slices at **186 FPS end to end on an RTX 5070 Ti** with a TensorRT fp16
engine (batch 16), and at **155 FPS with batch 1** using fp16, channels_last and `torch.compile`. Both are
more than twice the 70-80 FPS target. The knot masks of these variants match the fp32 reference with knot
pixel IoU 0.9994-0.9995 on log 10. A 45 W laptop GPU (RTX 1000 Ada) reaches at most 38.6 FPS, about half
the target.

## Target per GPU

| GPU | best e2e `fps_throughput` | variant (batch) | 70-80 FPS target |
|---|---|---|---|
| NVIDIA GeForce RTX 5070 Ti (euler, 16 GB, 300 W) | **185.9** | `trt_fp16` (16) | met, 2.3x the upper bound |
| NVIDIA GeForce RTX 5070 Ti, batch 1 | 155.3 | `fp16_cl_compile_cg` (1) | met, 1.9x |
| NVIDIA GeForce RTX 5070 Ti, no compilation | 129.4 | `fp16_cl` (4) | met, 1.6x |
| NVIDIA RTX 1000 Ada Generation Laptop GPU (6 GB, 45 W) | 38.6 | `fp16_cl_compile_cg` (2) | not met, 0.48x of 80 |
| GTX 1650 (Vostro) | not measured | | |

## Method

- Script: [scripts/bench_inference.py](../../scripts/bench_inference.py), run from the repository root:

  ```bash
  python -m scripts.bench_inference \
      --config src/configs/train_kwp_v4_unet_effv2s_w1_dice_ts_48ep_dec512.yaml \
      --checkpoint <path>/seg_head_epoch_47.pth \
      --img_dir <kwp_ds_v4>/10/img --host <name> --out <file>.json
  ```

- Input: log 10 of kwp-ds-v4 (299 slices, native 778 x 778), 3-slice stacks [z-1, z, z+1] as uint8 in
  pinned host memory. The model runs at its training resolution, 800 x 800.
- **e2e** (end to end): host-to-device copy, /255, bilinear 778 -> 800, ImageNet normalisation, U-Net,
  softmax, bilinear 800 -> 778, argmax, pith soft-argmax, device-to-host copy of the class map and the pith
  point. TIFF decoding is not included. **model**: the U-Net forward pass on a preprocessed GPU tensor.
- Timing: `torch.inference_mode`, `cudnn.benchmark`, 20 warm-up iterations (they include compilation and
  TensorRT engine builds), then per repeat 100 timed iterations with `torch.cuda.synchronize()` after each
  (latency) and 100 iterations synchronised once at the end (throughput). euler: 5 repeats; Precision: 50
  iterations x 3 repeats.
- Logged keys (in `results/*.json`, one entry per variant, batch size and mode):
  - `fps_throughput`: median over repeats of batch_size x iterations / wall time of the unsynchronised loop.
  - `fps_latency` = batch_size x 1000 / `latency_ms_mean`.
  - `latency_ms_p50`, `latency_ms_p95`: per-batch latency over all synchronised iterations.
  - `gpu_before`, `gpu_after`: `nvidia-smi` memory, utilisation, clocks and power around each measurement.
- Accuracy: every variant predicts all 299 slices of log 10 (batch 4) and is compared with the `fp32`
  variant (TF32 off) on the same GPU:
  - `knot_pixel_iou_vs_ref`: IoU of the knot masks (argmax == knot) pooled over the log.
  - `pith_px_diff_max`: largest distance between the pith points, in native pixels.
  - `class_pixels_differing`, `pith_rounded_points_differing`: in the JSON files.

### Variants

| variant | weights / activations | memory format | compilation |
|---|---|---|---|
| `fp32` | fp32, TF32 off (reference) | NCHW | none |
| `tf32` | fp32, TF32 convolutions | NCHW | none |
| `bf16_autocast` | fp32 weights, bf16 autocast (setting of the M5 exports) | NCHW | none |
| `fp16_autocast` | fp32 weights, fp16 autocast | NCHW | none |
| `fp16` | fp16 weights and input | NCHW | none |
| `fp16_cl` | fp16 | channels_last | none |
| `bf16_cl` | bf16 | channels_last | none |
| `fp16_cl_compile` | fp16 | channels_last | `torch.compile(mode="max-autotune-no-cudagraphs")` |
| `fp16_cl_compile_cg` | fp16 | channels_last | `torch.compile(mode="max-autotune")` (with CUDA graphs) |
| `trt_fp16` | fp16 | NCHW | TensorRT engine via torch-tensorrt (dynamo, optimisation level 5), one static engine per batch size |

Software: torch 2.7.0+cu128, cuDNN 9.7.1, segmentation-models-pytorch 0.5.0, torch-tensorrt 2.7.0
(TensorRT 10.9).

## Results

Best value per column in bold; for latency lower is better. Batch sizes 1, 2, 4, 8, 16.

### euler: NVIDIA GeForce RTX 5070 Ti

| variant | `knot_pixel_iou_vs_ref` | `pith_px_diff_max` | e2e `fps_throughput`, batch 1 | e2e `latency_ms_p50` / `p95`, batch 1 | best e2e `fps_throughput` (batch) | best model `fps_throughput` (batch) |
|---|---|---|---|---|---|---|
| fp32 | 1.0000 | 0.000 | 42.6 | 23.5 / 23.6 | 54.4 (16) | 55.4 (16) |
| tf32 | 0.9998 | 0.013 | 51.9 | 19.3 / 19.4 | 61.3 (4) | 62.6 (4) |
| bf16_autocast | 0.9973 | 0.101 | 50.0 | 20.0 / 20.1 | 90.1 (4) | 92.9 (4) |
| fp16_autocast | 0.9997 | 0.025 | 52.4 | 19.1 / 19.2 | 96.7 (4) | 100.0 (4) |
| fp16 | 0.9995 | 0.014 | 60.4 | 16.5 / 16.6 | 107.5 (4) | 111.6 (4) |
| fp16_cl | 0.9995 | 0.015 | 70.9 | 14.1 / 14.4 | 129.4 (4) | 135.7 (4) |
| bf16_cl | 0.9970 | 0.092 | 69.2 | 14.5 / 14.6 | 126.2 (4) | 131.9 (4) |
| fp16_cl_compile | 0.9995 | 0.025 | 109.6 | 9.1 / 9.2 | 168.4 (4) | 181.0 (4) |
| fp16_cl_compile_cg | 0.9995 | 0.025 | **155.3** | **6.5** / 6.5 | 171.6 (4) | 185.1 (2) |
| trt_fp16 | 0.9994 | 0.053 | 128.8 | 7.8 / 7.8 | **185.9** (16) | **200.2** (16) |

e2e `fps_throughput` per batch size:

| variant | b1 | b2 | b4 | b8 | b16 |
|---|---|---|---|---|---|
| fp32 | 42.6 | 47.1 | 48.4 | 47.7 | 54.4 |
| tf32 | 51.9 | 60.5 | 61.3 | 59.0 | 59.2 |
| bf16_autocast | 50.0 | 76.5 | 90.1 | 87.6 | 85.5 |
| fp16_autocast | 52.4 | 79.4 | 96.7 | 93.7 | 91.1 |
| fp16 | 60.4 | 92.7 | 107.5 | 103.8 | 100.9 |
| fp16_cl | 70.9 | 105.0 | 129.4 | 128.4 | 123.4 |
| bf16_cl | 69.2 | 104.7 | 126.2 | 125.1 | 120.6 |
| fp16_cl_compile | 109.6 | 158.8 | 168.4 | 165.0 | 160.0 |
| fp16_cl_compile_cg | 155.3 | 170.8 | 171.6 | 165.9 | 160.3 |
| trt_fp16 | 128.8 | 156.6 | 170.8 | 178.5 | 185.9 |

The euler GPU is shared with other users whose processes are not visible from the container. `fp32` and
`fp16_cl` were measured twice, in two runs about two hours apart (`results/euler_pass1.json`,
`results/euler_pass2.json`): e2e `fps_throughput` differs by at most 0.3 FPS. The 5 repeats of a
measurement differ by at most 4.6 % (one `fp16_cl` batch-1 repeat), most by well under 1 %. Before the first run `nvidia-smi` reported 1.4 GB used by others
and 0 % utilisation. The tables use the first run, and `trt_fp16` comes from the second.

### Precision: NVIDIA RTX 1000 Ada Generation Laptop GPU

| variant | `knot_pixel_iou_vs_ref` | `pith_px_diff_max` | e2e `fps_throughput`, batch 1 | e2e `latency_ms_p50` / `p95`, batch 1 | best e2e `fps_throughput` (batch) | best model `fps_throughput` (batch) |
|---|---|---|---|---|---|---|
| fp32 | 1.0000 | 0.000 | 13.8 | 72.9 / 74.2 | 13.8 (1) | 13.9 (1) |
| tf32 | 0.9999 | 0.017 | 18.4 | 54.5 / 55.8 | 18.4 (1) | 18.8 (1) |
| bf16_autocast | 0.9973 | 0.108 | 24.0 | 41.2 / 43.0 | 24.4 (2) | 26.1 (1) |
| fp16_autocast | 0.9997 | 0.025 | 25.2 | 39.6 / 41.1 | 26.2 (2) | 27.9 (1) |
| fp16 | 0.9995 | 0.015 | 30.3 | 33.0 / 34.2 | 30.3 (1) | 32.4 (1) |
| fp16_cl | 0.9995 | 0.015 | 33.9 | 29.3 / 32.6 | 35.8 (2) | 39.2 (1) |
| bf16_cl | 0.9970 | 0.104 | 34.3 | 29.4 / 30.5 | 35.2 (2) | 38.6 (1) |
| fp16_cl_compile | 0.9995 | 0.023 | 37.4 | **25.4** / 35.4 | 37.4 (1) | **44.5** (1) |
| fp16_cl_compile_cg | 0.9995 | 0.023 | **38.5** | 26.0 / 27.9 | **38.6** (2) | 42.2 (1) |
| trt_fp16 | 0.9993 | 0.038 | 31.2 | 30.8 / 33.3 | 35.6 (8) | 38.0 (16) |

e2e `fps_throughput` per batch size (OOM = out of the 6 GB GPU memory):

| variant | b1 | b2 | b4 | b8 | b16 |
|---|---|---|---|---|---|
| fp32 | 13.8 | 12.9 | 12.1 | 12.0 | OOM |
| tf32 | 18.4 | 16.2 | 15.3 | 14.9 | OOM |
| bf16_autocast | 24.0 | 24.4 | 23.2 | 22.3 | OOM |
| fp16_autocast | 25.2 | 26.2 | 24.6 | 23.7 | OOM |
| fp16 | 30.3 | 30.0 | 27.9 | 26.8 | 22.7 |
| fp16_cl | 33.9 | 35.8 | 33.3 | 31.5 | 30.5 |
| bf16_cl | 34.3 | 35.2 | 33.2 | 31.6 | 30.0 |
| fp16_cl_compile | 37.4 | 33.1 | 34.6 | 35.3 | 34.3 |
| fp16_cl_compile_cg | 38.5 | 38.6 | 37.9 | 35.3 | 34.2 |
| trt_fp16 | 31.2 | 33.7 | 29.7 | 35.6 | 32.8 |

On this GPU throughput does not grow with batch size, and repeats of one measurement differ by up to
54 % (`fp16_cl_compile` model, batch 8: 22.9, 23.5, 35.6), so differences of a few FPS between variants are
within noise here. During the runs another job was using CPU and RAM on the machine (not the GPU). `trt_fp16` comes from a separate run
(`results/precision_trt.json`), because building TensorRT engines for several batch sizes in one process ran
out of GPU memory.

## Observations

- Half precision is the largest single gain: on the RTX 5070 Ti `fp16` doubles the best fp32 throughput
  (107.5 vs 54.4), channels_last adds 20 % (129.4), and `torch.compile` another 30 % (168.4).
- fp16 stays closer to the fp32 reference than bf16: knot IoU 0.9995 vs 0.9970-0.9973 and pith up to 0.015
  vs 0.10 px. The M5 pre-label exports use bf16 autocast.
- At batch 1, compilation with CUDA graphs (`fp16_cl_compile_cg`) is the fastest: 6.5 ms per slice. TensorRT
  is fastest at large batches (185.9 FPS at batch 16), where latency per batch is 86 ms.
- Compilation cost (in `warmup_s`): `torch.compile` with max-autotune took up to 2 minutes per batch size on
  euler and 3 on Precision; a TensorRT engine took up to 8.5 minutes per batch size on euler and up to 22 on
  Precision. Both have to be built
  once per GPU and input shape.
- The e2e pre/post-processing (copies, resizes, softmax, argmax) lowers the best throughput by 2-7 %
  relative to the model alone on the RTX 5070 Ti (1 - best e2e `fps_throughput` / best model `fps_throughput`).
