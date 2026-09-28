from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from classical_segmentation import mask_dice, mask_iou

MODEL_NAME = "facebook/sam2.1-hiera-tiny"


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw the SAM2 mask and its contour on top of the original image."""
    overlay = image.copy()
    foreground = mask > 0

    # Use a light transparent fill so the person is still visible underneath.
    tint = np.zeros_like(image)
    tint[foreground] = (0, 255, 0)
    overlay = cv2.addWeighted(overlay, 0.70, tint, 0.30, 0)

    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    return overlay


def save_comparison_panel(
    image: np.ndarray,
    classical_mask: np.ndarray,
    sam2_mask: np.ndarray,
    output_path: Path,
) -> None:
    """Save a simple side-by-side visual comparison."""
    classical_vis = cv2.cvtColor(classical_mask, cv2.COLOR_GRAY2BGR)
    sam2_vis = cv2.cvtColor(sam2_mask, cv2.COLOR_GRAY2BGR)
    original = image.copy()
    sam2_overlay = make_overlay(image, sam2_mask)

    h, w = image.shape[:2]
    classical_vis = cv2.resize(classical_vis, (w, h))
    sam2_vis = cv2.resize(sam2_vis, (w, h))

    panel = np.hstack([original, classical_vis, sam2_vis, sam2_overlay])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), panel)


def run_sam2(
    image_path: Path,
    bbox: tuple[int, int, int, int],
    output_mask: Path,
) -> np.ndarray:
    """Run SAM2 with the supplied bounding-box prompt and save the mask."""
    try:
        import torch
        from transformers import Sam2Model, Sam2Processor
    except ImportError as exc:
        raise RuntimeError(
            "SAM2 dependencies are not installed. Install requirements_sam2.txt first."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Sam2Model.from_pretrained(MODEL_NAME).to(device)
    processor = Sam2Processor.from_pretrained(MODEL_NAME)

    image = Image.open(image_path).convert("RGB")
    x1, y1, x2, y2 = map(int, bbox)

    inputs = processor(
        images=image,
        input_boxes=[[[x1, y1, x2, y2]]],
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)

    masks = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu()
    )
    mask = masks[0][0][0].numpy().astype(np.uint8) * 255

    output_mask.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_mask), mask)
    return mask


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the classical mask with SAM2."
    )
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument(
        "--bbox",
        nargs=4,
        required=True,
        type=int,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="The same person box used for the classical method.",
    )
    parser.add_argument("--classical-mask", required=True, type=Path)
    parser.add_argument("--output-mask", required=True, type=Path)
    args = parser.parse_args()

    bbox = tuple(args.bbox)
    sam2_mask = run_sam2(args.image, bbox, args.output_mask)

    classical_mask = cv2.imread(str(args.classical_mask), cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(str(args.image), cv2.IMREAD_COLOR)

    if classical_mask is None:
        raise FileNotFoundError(
            f"Could not read classical mask: {args.classical_mask}"
        )
    if image is None:
        raise FileNotFoundError(f"Could not read input image: {args.image}")

    iou = mask_iou(classical_mask, sam2_mask)
    dice = mask_dice(classical_mask, sam2_mask)

    output_dir = args.output_mask.parent
    overlay_path = output_dir / "sam2_overlay.png"
    comparison_path = output_dir / "comparison_panel.png"
    metrics_path = output_dir / "comparison_metrics.json"

    cv2.imwrite(str(overlay_path), make_overlay(image, sam2_mask))
    save_comparison_panel(image, classical_mask, sam2_mask, comparison_path)

    metrics = {
        "model": MODEL_NAME,
        "bbox": list(map(int, bbox)),
        "iou": float(iou),
        "dice": float(dice),
        "note": "IoU and Dice measure agreement between the classical mask and the SAM2 mask, not ground-truth accuracy.",
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"SAM2 model: {MODEL_NAME}")
    print(f"IoU (classical vs SAM2):  {iou:.4f}")
    print(f"Dice (classical vs SAM2): {dice:.4f}")
    print(f"SAM2 overlay: {overlay_path}")
    print(f"Comparison panel: {comparison_path}")
    print(f"Comparison metrics: {metrics_path}")


if __name__ == "__main__":
    main()
