from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SegmentationResult:
    mask: np.ndarray
    overlay: np.ndarray
    contour: np.ndarray | None
    bounding_box: tuple[int, int, int, int]
    area_pixels: int
    perimeter_pixels: float


def _clip_box(box: tuple[int, int, int, int], width: int, height: int) -> tuple[int, int, int, int]:
    """Keep the user-provided box inside the image."""
    x1, y1, x2, y2 = box
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(x1 + 1, min(width, int(x2)))
    y2 = max(y1 + 1, min(height, int(y2)))
    return x1, y1, x2, y2


def _largest_component(mask: np.ndarray, min_area: int = 100) -> np.ndarray:
    """Keep the main connected part of the mask."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

    if count <= 1:
        return binary * 255

    best_label = 0
    best_area = 0
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area and area > best_area:
            best_label = label
            best_area = area

    if best_label == 0:
        best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))

    return (labels == best_label).astype(np.uint8) * 255


def _postprocess(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Clean up the mask and put it back into full-image coordinates."""
    height, width = image_shape[:2]
    x1, y1, x2, y2 = _clip_box(bbox, width, height)
    full = np.zeros((height, width), np.uint8)
    roi = (mask > 0).astype(np.uint8) * 255

    if roi.size == 0:
        return full

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    roi = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, kernel, iterations=2)
    roi = cv2.morphologyEx(roi, cv2.MORPH_OPEN, kernel, iterations=1)

    # Fill small holes from the outside of the mask.
    padded = cv2.copyMakeBorder(roi, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (1, 1), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    roi = cv2.bitwise_or(roi, holes)

    min_area = max(100, int(0.01 * roi.shape[0] * roi.shape[1]))
    roi = _largest_component(roi, min_area)
    full[y1:y2, x1:x2] = roi
    return full


def _roi_border_lab(roi_lab: np.ndarray, border: int = 6) -> np.ndarray:
    """Estimate the background color from the edges of the ROI."""
    h, w = roi_lab.shape[:2]
    b = max(2, min(border, h // 8, w // 8))
    pixels = np.concatenate(
        [
            roi_lab[:b].reshape(-1, 3),
            roi_lab[-b:].reshape(-1, 3),
            roi_lab[:, :b].reshape(-1, 3),
            roi_lab[:, -b:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(pixels, axis=0)


def segment_rgb(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Segment a person in an RGB image using classical OpenCV operations."""
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_box(bbox, width, height)
    roi = image_bgr[y1:y2, x1:x2]

    # Compare each pixel with the background around the ROI.
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    background = _roi_border_lab(lab)
    color_distance = np.linalg.norm(lab - background[None, None, :], axis=2)
    color_distance = cv2.normalize(
        color_distance, None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    _, color_mask = cv2.threshold(
        color_distance, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Add edges so that clothing and body boundaries are easier to keep.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 40, 120)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    edges = cv2.dilate(edges, edge_kernel, iterations=1)

    candidate = cv2.bitwise_or(color_mask, edges)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    candidate = cv2.morphologyEx(
        candidate, cv2.MORPH_CLOSE, close_kernel, iterations=2
    )

    # A distance transform gives us a more reliable foreground core.
    dist = cv2.distanceTransform(
        (candidate > 0).astype(np.uint8), cv2.DIST_L2, 5
    )
    if float(dist.max()) > 0:
        sure_foreground = (
            dist > 0.35 * dist.max()
        ).astype(np.uint8) * 255
    else:
        sure_foreground = np.zeros_like(candidate)

    return _postprocess(
        cv2.bitwise_or(candidate, sure_foreground),
        bbox,
        image_bgr.shape,
    )


def segment_thermal(image_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Use the thermal image intensity to separate the person from the background."""
    height, width = image_bgr.shape[:2]
    x1, y1, x2, y2 = _clip_box(bbox, width, height)
    roi = image_bgr[y1:y2, x1:x2]

    # The downloaded thermal image uses a false-color scale, but the person
    # is still noticeably brighter than most of the dark background.
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu gives the cutoff from the image itself instead of using a fixed
    # thermal value. This makes the simple method a little less fragile.
    otsu_value, hot = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Clean small gaps in the person's silhouette while keeping the arm and
    # hand connected to the rest of the body.
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    hot = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    hot = cv2.morphologyEx(hot, cv2.MORPH_OPEN, open_kernel, iterations=1)

    # Keep the largest connected region inside the user-provided box.
    # For this image that region is the person rather than the dark water/background.
    hot = _largest_component(hot, min_area=max(50, int(0.01 * hot.size)))

    # If Otsu somehow produces a very small region, fall back to a high-
    # intensity threshold based on the ROI percentiles.
    min_reasonable = max(100, int(0.05 * hot.size))
    if cv2.countNonZero(hot) < min_reasonable:
        cutoff = float(np.percentile(gray, 75.0))
        _, fallback = cv2.threshold(gray, cutoff, 255, cv2.THRESH_BINARY)
        fallback = cv2.morphologyEx(
            fallback, cv2.MORPH_CLOSE, close_kernel, iterations=2
        )
        hot = _largest_component(
            fallback, min_area=max(50, int(0.01 * fallback.size))
        )

    return _postprocess(
        hot,
        bbox,
        image_bgr.shape,
    )


def build_result(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> SegmentationResult:
    """Build the mask overlay and a few simple measurements."""
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contour = max(contours, key=cv2.contourArea) if contours else None
    area = int(cv2.countNonZero(mask))
    perimeter = float(cv2.arcLength(contour, True)) if contour is not None else 0.0

    overlay = image_bgr.copy()
    green = np.zeros_like(image_bgr)
    green[:, :, 1] = mask
    overlay = cv2.addWeighted(overlay, 0.72, green, 0.28, 0)

    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

    x1, y1, x2, y2 = bbox
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return SegmentationResult(
        mask,
        overlay,
        contour,
        bbox,
        area,
        perimeter,
    )


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    intersection = float(np.logical_and(a, b).sum())
    union = float(np.logical_or(a, b).sum())
    return intersection / union if union else 1.0


def mask_dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a > 0
    b = mask_b > 0
    intersection = float(np.logical_and(a, b).sum())
    total = float(a.sum() + b.sum())
    return (2 * intersection / total) if total else 1.0


def fourier_edge_map(gray: np.ndarray, cutoff: float = 0.12) -> np.ndarray:
    """Make a simple high-pass edge image using the FFT."""
    gray_f = gray.astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray_f))

    h, w = gray.shape
    yy, xx = np.ogrid[:h, :w]
    cy, cx = h / 2.0, w / 2.0
    radius = np.sqrt(
        ((yy - cy) / max(cy, 1.0)) ** 2
        + ((xx - cx) / max(cx, 1.0)) ** 2
    )

    highpass = (radius >= cutoff).astype(np.float32)
    filtered = spectrum * highpass
    result = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))
    result = np.abs(result)
    return cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
