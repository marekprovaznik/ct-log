import json
import math
import sys

import numpy as np
import slicer
from vtk.util.numpy_support import vtk_to_numpy

CT = "/mnt/D/datasets/ct_log/slicer/log10/log10_ct.nrrd"
SEG = "/mnt/D/datasets/ct_log/slicer/log10/log10_labels.seg.nrrd"
OUT = sys.argv[-1] if sys.argv[-1].endswith(".json") else "/mnt/D/datasets/ct_log/radial/log10/radial/slicer_check.json"
SHOTS = OUT.replace(".json", "")
CASES = json.loads(open(OUT.replace(".json", "_cases.json")).read())


def main() -> None:
    ct = slicer.util.loadVolume(CT)
    seg = slicer.util.loadSegmentation(SEG)
    seg.SetReferenceImageGeometryParameterFromVolumeNode(ct)
    slicer.util.setSliceViewerLayers(background=ct, fit=False)
    lm = slicer.app.layoutManager()
    lm.setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUpRedSliceView)
    widget = lm.sliceWidget("Red")
    logic = widget.sliceLogic()
    node = logic.GetSliceNode()
    results = []
    for case in CASES:
        width = int(case["width"])
        n_rows = int(case["n_rows"])
        theta = math.radians(case["theta_deg"])
        px, py = case["pith_xy"]
        u = np.array([-math.cos(theta), -math.sin(theta), 0.0])
        v = np.array([0.0, 0.0, 1.0])
        n = np.cross(u, v)
        origin = np.array([-px, -py, 10.0 * (n_rows - 1) / 2.0])
        m = node.GetSliceToRAS()
        for i in range(3):
            m.SetElement(i, 0, u[i])
            m.SetElement(i, 1, v[i])
            m.SetElement(i, 2, n[i])
            m.SetElement(i, 3, origin[i])
        node.SetDimensions(width, n_rows, 1)
        node.SetFieldOfView(float(width), 10.0 * n_rows, 1.0)
        node.UpdateMatrices()
        slicer.app.processEvents()
        reslice = logic.GetBackgroundLayer().GetReslice()
        reslice.Update()
        img = reslice.GetOutput()
        dims = img.GetDimensions()
        arr = vtk_to_numpy(img.GetPointData().GetScalars()).reshape(dims[1], dims[0], -1)[..., 0]
        np.save("%s_%s.npy" % (SHOTS, case["name"]), arr)
        xy = node.GetXYToRAS()
        np.save(
            "%s_%s_xytoras.npy" % (SHOTS, case["name"]),
            np.array([[xy.GetElement(i, j) for j in range(4)] for i in range(4)]),
        )
        widget.sliceView().forceRender()
        slicer.util.forceRenderAllViews()
        cap = slicer.util.mainWindow().centralWidget().grab()
        cap.save("%s_%s.png" % (SHOTS, case["name"]))
        results.append({"name": case["name"], "shape": list(arr.shape)})
    json.dump(results, open(OUT, "w"), indent=1)
    slicer.util.exit(0)


main()
