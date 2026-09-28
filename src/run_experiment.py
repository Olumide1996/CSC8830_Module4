from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from classical_segmentation import build_result, fourier_edge_map, segment_rgb, segment_thermal


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Module 4 segmentation experiment.")
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--modality", choices=["rgb", "thermal"], required=True)
    parser.add_argument(
        "--bbox",
        nargs=4,
        required=True,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Approximate person box: left top right bottom.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    bbox = tuple(args.bbox)
    segmenter = segment_rgb if args.modality == "rgb" else segment_thermal

    mask = segmenter(image, bbox)
    result = build_result(image, mask, bbox)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(args.output_dir / "input.png"), image)
    cv2.imwrite(str(args.output_dir / "classical_mask.png"), result.mask)
    cv2.imwrite(str(args.output_dir / "classical_overlay.png"), result.overlay)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(
        str(args.output_dir / "fourier_edges.png"),
        fourier_edge_map(gray),
    )

    metrics = {
        "modality": args.modality,
        "image": str(args.image),
        "bbox": list(map(int, bbox)),
        "area_pixels": result.area_pixels,
        "perimeter_pixels": result.perimeter_pixels,
        "mask_fraction": float(result.area_pixels / result.mask.size),
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print(f"Modality: {args.modality}")
    print(f"Image: {args.image}")
    print(f"Bounding box: {bbox}")
    print(f"Foreground pixels: {result.area_pixels}")
    print(f"Mask fraction: {metrics['mask_fraction']:.4f}")
    print(f"Perimeter: {result.perimeter_pixels:.1f} px")
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
