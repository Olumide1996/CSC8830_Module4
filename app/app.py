from __future__ import annotations

import json
from pathlib import Path
import sys

import cv2
import numpy as np
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classical_segmentation import build_result, fourier_edge_map, segment_rgb, segment_thermal


st.set_page_config(page_title="CSc 8830 Module 4", page_icon="🎯", layout="wide")

st.title("CSc 8830 Module 4: Human Boundary Segmentation")
st.caption("Classical OpenCV segmentation for RGB and thermal images, with a SAM2 comparison and Fourier-domain analysis.")
st.info(
    "The main RGB and thermal segmentation methods use traditional OpenCV image processing. "
    "SAM2 is used separately as a reference comparison, not as part of the main classical pipeline."
)


def read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def show_completed_example(modality: str, title: str) -> None:
    output_dir = ROOT / "outputs" / modality
    st.header(title)

    required = output_dir / "classical_overlay.png"
    if not required.exists():
        st.warning(
            f"Saved {modality.upper()} results are not available yet. "
            "Run the experiment script first."
        )
        return

    metrics = read_json(output_dir / "metrics.json") or {}
    comparison = read_json(output_dir / "comparison_metrics.json") or {}

    st.subheader("Classical OpenCV result")
    c1, c2 = st.columns(2)
    with c1:
        input_path = output_dir / "input.png"
        if input_path.exists():
            st.image(str(input_path), caption="Input image", width="stretch")
    with c2:
        st.image(
            str(output_dir / "classical_overlay.png"),
            caption="Classical OpenCV human boundary",
            width="stretch",
        )

    mask_path = output_dir / "classical_mask.png"
    if mask_path.exists():
        st.image(str(mask_path), caption="Classical binary mask", width="stretch")

    if metrics:
        m1, m2, m3 = st.columns(3)
        if "area_pixels" in metrics:
            m1.metric("Foreground pixels", f"{metrics['area_pixels']:,}")
        if "mask_fraction" in metrics:
            m2.metric("Mask fraction", f"{metrics['mask_fraction']:.2%}")
        if "perimeter_pixels" in metrics:
            m3.metric("Boundary length", f"{metrics['perimeter_pixels']:.1f} px")

    st.subheader("SAM2 comparison")
    sam2_mask = output_dir / "sam2_mask.png"
    sam2_overlay = output_dir / "sam2_overlay.png"
    panel = output_dir / "comparison_panel.png"

    if panel.exists():
        st.image(
            str(panel),
            caption="Original image, classical mask, SAM2 mask, and SAM2 overlay",
            width="stretch",
        )
    else:
        cols = st.columns(2)
        if sam2_mask.exists():
            with cols[0]:
                st.image(str(sam2_mask), caption="SAM2 mask", width="stretch")
        if sam2_overlay.exists():
            with cols[1]:
                st.image(str(sam2_overlay), caption="SAM2 overlay", width="stretch")

    if comparison:
        m1, m2 = st.columns(2)
        m1.metric("Classical vs SAM2 IoU", f"{comparison['iou']:.4f}")
        m2.metric("Classical vs SAM2 Dice", f"{comparison['dice']:.4f}")
        st.caption(
            "IoU and Dice here measure agreement between the classical segmentation mask and the SAM2 mask. "
            "They are not ground-truth accuracy measurements."
        )
    else:
        st.info("SAM2 comparison outputs have not been generated for this modality yet.")

    edge_path = output_dir / "fourier_edges.png"
    if edge_path.exists():
        st.subheader("Fourier edge analysis")
        st.image(
            str(edge_path),
            caption="High-frequency edge response",
            width="stretch",
        )


show_completed_example("rgb", "1. RGB completed example")
show_completed_example("thermal", "2. Thermal completed example")

st.header("3. Fourier-domain demonstration")
fourier_dir = ROOT / "outputs" / "rgb" / "fourier"

if fourier_dir.exists():
    f1, f2 = st.columns(2)
    with f1:
        path = fourier_dir / "fourier_spectrum.png"
        if path.exists():
            st.image(str(path), caption="Centered Fourier magnitude spectrum", width="stretch")
    with f2:
        path = fourier_dir / "fourier_edges.png"
        if path.exists():
            st.image(str(path), caption="Frequency-domain edge map", width="stretch")

    f3, f4 = st.columns(2)
    with f3:
        path = fourier_dir / "fourier_low_pass.png"
        if path.exists():
            st.image(str(path), caption="Low-pass filtered image", width="stretch")
    with f4:
        path = fourier_dir / "fourier_segmentation_overlay.png"
        if path.exists():
            st.image(
                str(path),
                caption="Threshold-based frequency-domain region demonstration",
                width="stretch",
            )

    st.warning(
        "The simple frequency-filter/threshold demonstration over-segments strong background edges. "
        "It is included to illustrate the theory, not as the final human segmentation method."
    )
else:
    st.info("Run scripts/fourier_analysis.py to generate the Fourier demonstration outputs.")

st.header("4. Run the classical method yourself")
modality = st.selectbox("Image type", ["RGB", "Thermal"])
uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "bmp"],
)

if uploaded is not None:
    data = np.frombuffer(uploaded.getvalue(), np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        st.error("The image could not be read.")
        st.stop()

    height, width = image.shape[:2]
    st.caption("Give a rough box around the person. The classical method then refines the boundary inside that region.")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        x1 = st.number_input("x1", 0, max(0, width - 2), int(0.20 * width), 1)
    with col2:
        y1 = st.number_input("y1", 0, max(0, height - 2), int(0.10 * height), 1)
    with col3:
        x2 = st.number_input("x2", 1, width, int(0.80 * width), 1)
    with col4:
        y2 = st.number_input("y2", 1, height, int(0.95 * height), 1)

    if x2 <= x1 or y2 <= y1:
        st.error("Please make sure x2 is greater than x1 and y2 is greater than y1.")
        st.stop()

    bbox = (int(x1), int(y1), int(x2), int(y2))
    segmenter = segment_rgb if modality == "RGB" else segment_thermal
    mask = segmenter(image, bbox)
    result = build_result(image, mask, bbox)

    c1, c2 = st.columns(2)
    with c1:
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Input",
            width="stretch",
        )
    with c2:
        st.image(
            cv2.cvtColor(result.overlay, cv2.COLOR_BGR2RGB),
            caption="Classical segmentation",
            width="stretch",
        )

    st.subheader("Boundary details")
    m1, m2, m3 = st.columns(3)
    m1.metric("Foreground pixels", f"{result.area_pixels:,}")
    m2.metric("Mask fraction", f"{result.area_pixels / (width * height):.2%}")
    m3.metric("Boundary length", f"{result.perimeter_pixels:.1f} px")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.image(
        fourier_edge_map(gray),
        caption="Fourier high-pass edge map for the uploaded image",
        width="stretch",
    )

st.header("5. Theory")

st.subheader("Edge detection in the frequency domain")
st.markdown(
    "Let the image be represented by f(x,y), with Fourier transform F(u,v). "
    "Differentiation in the spatial domain becomes multiplication by frequency in the Fourier domain:"
)
st.latex(r"\mathcal{F}\left\{\frac{\partial f}{\partial x}\right\}=j2\pi uF(u,v), \qquad \mathcal{F}\left\{\frac{\partial f}{\partial y}\right\}=j2\pi vF(u,v).")

st.markdown(
    "A gradient magnitude can then be formed after transforming the derivative responses back to the spatial domain:"
)
st.latex(r"G(x,y)=\sqrt{G_x^2(x,y)+G_y^2(x,y)}.")

st.markdown(
    "Large responses occur where image intensity changes rapidly, which is why the method highlights object boundaries as well as strong background edges."
)

st.markdown("For the Laplacian, the Fourier relationship is:")
st.latex(r"\mathcal{F}\{\nabla^2 f\}=-4\pi^2(u^2+v^2)F(u,v).")
st.markdown("This also emphasizes higher-frequency content.")

st.subheader("Segmentation in the frequency domain")
st.markdown(
    "Frequency filtering changes the relative contribution of different spatial scales. For a filter H(u,v):"
)
st.latex(r"g(x,y)=\mathcal{F}^{-1}\{H(u,v)F(u,v)\}.")

st.markdown("A simple region mask can then be produced with a threshold T:")
st.latex(r"M(x,y)=\begin{cases}1,&g(x,y)>T\\0,&g(x,y)\le T.\end{cases}")

st.markdown(
    "In this project, the Fourier demonstration shows that this basic idea can separate some intensity structures, "
    "but it also responds to strong background edges. The final RGB and thermal human masks therefore use "
    "classical OpenCV spatial-domain segmentation, while Fourier analysis is included as the requested frequency-domain study."
)

st.header("6. SAM2 note")
st.write(
    "SAM2 is used only for comparison. The main segmentation scripts do not call a deep-learning or machine-learning model. "
    "The saved SAM2 results were generated separately with facebook/sam2.1-hiera-tiny using the same approximate bounding-box prompt."
)
