import cv2
import numpy as np

from src.classical_segmentation import (
    build_result,
    fourier_edge_map,
    mask_dice,
    mask_iou,
    segment_rgb,
    segment_thermal,
)


def synthetic_image():
    """Make a simple person-like image so the tests do not need a real photo."""
    image = np.full((220, 180, 3), 210, np.uint8)
    cv2.circle(image, (90, 45), 22, (175, 175, 175), -1)
    cv2.rectangle(image, (65, 65), (115, 135), (60, 120, 220), -1)
    cv2.rectangle(image, (72, 132), (88, 200), (80, 80, 80), -1)
    cv2.rectangle(image, (92, 132), (108, 200), (80, 80, 80), -1)
    return image


def test_rgb_segmentation_returns_mask():
    image = synthetic_image()
    mask = segment_rgb(image, (45, 15, 135, 210))
    assert mask.shape == image.shape[:2]
    assert int(mask.sum()) > 0


def test_thermal_segmentation_returns_mask():
    image = np.zeros((200, 160, 3), np.uint8)
    cv2.rectangle(image, (60, 40), (105, 165), (240, 240, 240), -1)
    mask = segment_thermal(image, (35, 20, 125, 185))
    assert mask.shape == image.shape[:2]
    assert int(mask.sum()) > 0


def test_mask_metrics_identity():
    mask = np.zeros((20, 20), np.uint8)
    cv2.rectangle(mask, (5, 5), (12, 12), 255, -1)
    assert mask_iou(mask, mask) == 1.0
    assert mask_dice(mask, mask) == 1.0


def test_fourier_edge_map_shape():
    image = np.zeros((64, 64), np.uint8)
    image[:, 32:] = 255
    edges = fourier_edge_map(image)
    assert edges.shape == image.shape
    assert edges.dtype == np.uint8


def test_build_result_metrics():
    image = synthetic_image()
    mask = segment_rgb(image, (45, 15, 135, 210))
    result = build_result(image, mask, (45, 15, 135, 210))
    assert result.area_pixels > 0
    assert result.overlay.shape == image.shape
