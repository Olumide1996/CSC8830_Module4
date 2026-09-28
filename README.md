# CSc 8830 Computer Vision — Module 4

## Assignment
Implement classical human-boundary segmentation for (1) RGB images and (2) thermal images using OpenCV only for the main method. No deep learning or machine-learning model is used in the primary segmentation pipeline. Compare the classical boundaries with SAM2 and provide Fourier-domain theory for edge detection and region segmentation.

## Public example images
The project uses two public Wikimedia Commons examples:

1. RGB: `Man Standing.jpg` by Visitor7, CC BY-SA 3.0.
   Source: https://commons.wikimedia.org/wiki/File:Man_Standing.jpg
2. Thermal: `Man in water - IR image.jpg` by Krzysztof Jakucy, CC BY-SA 3.0; captured with a FLIR P640.
   Source: https://commons.wikimedia.org/wiki/File:Man_in_water_-_IR_image.jpg

Run:

```powershell
python scripts\download_examples.py
```

## Main method
The primary method is classical OpenCV processing. It uses a user-supplied bounding box to define the human region, then applies color/intensity separation, edges, morphology, connected components, and contour extraction. No neural network or learned classifier is used.

A Fourier high-pass edge map is also produced to support the theory portion.

## SAM2 comparison
SAM2 is used only as the comparison/reference method. The recommended comparison uses the same bounding-box prompt for both methods, then reports mask IoU and Dice overlap.

Install the optional comparison dependencies:

```powershell
pip install -r requirements_sam2.txt
```

Then use `src/sam2_compare.py` with the same image and bounding box used for the classical result.

## Run the classical experiment

RGB example:

```powershell
python src\run_experiment.py --image data\rgb\rgb_human.jpg --modality rgb --bbox 300 80 720 930 --output-dir outputs\rgb
```

Thermal example:

```powershell
python src\run_experiment.py --image data\thermal\thermal_human.jpg --modality thermal --bbox 190 150 450 340 --output-dir outputs\thermal
```

The bounding boxes are starting points. Check the actual image and adjust them if needed.

Run the web app:

```powershell
streamlit run app\app.py
```

Run tests:

```powershell
python -m pytest -q
```

## Important interpretation
IoU/Dice in the SAM2 comparison measure agreement between the two masks. They are not a ground-truth accuracy score. A human-verified ground-truth mask would be needed for true segmentation accuracy.
