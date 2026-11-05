from pathlib import Path

from camera_slider.tape_calibration import measure_pass
import cv2
import numpy as np
import pytest


def test_measure_pass_recovers_mm_per_step(tmp_path: Path) -> None:
    """Red labels every 100 mm on a yellow tape give back the mm per marker step used to draw them."""
    px_per_mm, mm_per_step = 2.0, 47.5
    image = np.zeros((200, 9200, 3), np.uint8)
    image[60:140] = (40, 200, 220)
    for k in range(40):
        x = int(round(100 + k * 100 * px_per_mm))
        image[80:120, x - 12 : x + 12] = (30, 30, 210)
    path = tmp_path / "render.jpg"
    cv2.imwrite(str(path), image)
    result = measure_pass(path, origin_x=3.0, px_per_step=px_per_mm * mm_per_step)
    assert result["labels"] == 40
    assert result["mm_per_step"] == pytest.approx(mm_per_step, rel=1e-3)
