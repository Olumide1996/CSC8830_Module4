from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def fft_shifted(image: np.ndarray) -> np.ndarray:
    """Return the centered 2-D Fourier transform."""
    spectrum = np.fft.fft2(image)
    return np.fft.fftshift(spectrum)


def magnitude_spectrum(frequency: np.ndarray) -> np.ndarray:
    """Convert a Fourier spectrum to a displayable magnitude image."""
    magnitude = np.log1p(np.abs(frequency))
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return magnitude.astype(np.uint8)


def make_frequency_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Create centered frequency coordinates in cycles/pixel."""
    rows, cols = shape
    fy = np.fft.fftshift(np.fft.fftfreq(rows))
    fx = np.fft.fftshift(np.fft.fftfreq(cols))
    u, v = np.meshgrid(fx, fy)
    return u, v


def fourier_gradient_edges(gray: np.ndarray) -> np.ndarray:
    """Compute a gradient-magnitude edge map using Fourier derivatives."""
    frequency = fft_shifted(gray.astype(np.float32))
    u, v = make_frequency_grid(gray.shape)

    # With numpy's frequency units, the derivative multipliers include 2*pi.
    fx = (1j * 2.0 * np.pi * u) * frequency
    fy = (1j * 2.0 * np.pi * v) * frequency

    gx = np.real(np.fft.ifft2(np.fft.ifftshift(fx)))
    gy = np.real(np.fft.ifft2(np.fft.ifftshift(fy)))
    magnitude = np.sqrt(gx * gx + gy * gy)

    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def gaussian_low_pass(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """Build a centered Gaussian low-pass filter."""
    u, v = make_frequency_grid(shape)
    radius_sq = u * u + v * v
    h = np.exp(-(radius_sq) / (2.0 * sigma * sigma))
    return h.astype(np.float32)


def fourier_low_pass(gray: np.ndarray, sigma: float) -> np.ndarray:
    """Low-pass filter an image in the Fourier domain and return uint8 output."""
    frequency = fft_shifted(gray.astype(np.float32))
    h = gaussian_low_pass(gray.shape, sigma)
    filtered = frequency * h
    spatial = np.real(np.fft.ifft2(np.fft.ifftshift(filtered)))
    spatial = cv2.normalize(spatial, None, 0, 255, cv2.NORM_MINMAX)
    return spatial.astype(np.uint8)


def threshold_region(image: np.ndarray) -> np.ndarray:
    """Threshold a filtered image and clean the mask with morphology."""
    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Keep the largest connected foreground component.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = np.where(labels == largest, 255, 0).astype(np.uint8)

    return mask


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw a green mask overlay on the source image."""
    overlay = image.copy()
    color = np.zeros_like(image)
    color[:, :, 1] = 255
    mask_bool = mask > 0
    overlay[mask_bool] = cv2.addWeighted(
        image[mask_bool], 0.45, color[mask_bool], 0.55, 0
    )
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fourier-domain edge detection and region segmentation demonstration."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.08,
        help="Gaussian low-pass bandwidth in normalized frequency units.",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    frequency = fft_shifted(gray.astype(np.float32))
    spectrum = magnitude_spectrum(frequency)
    edges = fourier_gradient_edges(gray)
    low_pass = fourier_low_pass(gray, args.sigma)
    mask = threshold_region(low_pass)
    overlay = make_overlay(image, mask)

    cv2.imwrite(str(args.output_dir / "fourier_spectrum.png"), spectrum)
    cv2.imwrite(str(args.output_dir / "fourier_edges.png"), edges)
    cv2.imwrite(str(args.output_dir / "fourier_low_pass.png"), low_pass)
    cv2.imwrite(str(args.output_dir / "fourier_segmentation_mask.png"), mask)
    cv2.imwrite(str(args.output_dir / "fourier_segmentation_overlay.png"), overlay)

    print(f"Input image: {args.image}")
    print(f"Low-pass sigma: {args.sigma}")
    print(f"Edge map: {args.output_dir / 'fourier_edges.png'}")
    print(f"Spectrum: {args.output_dir / 'fourier_spectrum.png'}")
    print(f"Low-pass result: {args.output_dir / 'fourier_low_pass.png'}")
    print(f"Segmentation mask: {args.output_dir / 'fourier_segmentation_mask.png'}")
    print(f"Segmentation overlay: {args.output_dir / 'fourier_segmentation_overlay.png'}")


if __name__ == "__main__":
    main()
