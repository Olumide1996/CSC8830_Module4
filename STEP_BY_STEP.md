# Module 4 — Beginner Step-by-Step Guide

## What this assignment asks for

1. Find a human boundary in an RGB image using classical OpenCV methods. Do not use machine learning for the main method.
2. Find a human boundary in a thermal image using classical OpenCV methods. Do not use machine learning for the main method.
3. Compare both results with SAM2.
4. Explain mathematically how Fourier-domain analysis can help with edge detection and segmentation.
5. Provide GitHub, a working web application, a PDF report, and a screen recording.

## Phase 1 — Set up the project

1. Extract the project ZIP.
2. Open `CSc8830_Module4_solution` in VS Code.
3. Open Terminal → New Terminal.
4. Create the environment:

```powershell
python -m venv .venv
```

5. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

6. Install the main packages:

```powershell
pip install -r requirements.txt
```

## Phase 2 — Download the two example images

Run:

```powershell
python scripts\download_examples.py
```

This downloads the RGB and thermal examples described in README.md.

## Phase 3 — Run the RGB experiment

Start with the suggested box:

```powershell
python src\run_experiment.py --image data\rgb\rgb_human.jpg --modality rgb --bbox 300 80 720 930 --output-dir outputs\rgb
```

If the overlay is not following the person well, change the four box numbers. The box is only a guide for the classical algorithm.

## Phase 4 — Run the thermal experiment

Start with:

```powershell
python src\run_experiment.py --image data\thermal\thermal_human.jpg --modality thermal --bbox 190 150 450 340 --output-dir outputs\thermal
```

Again, adjust the box if necessary.

## Phase 5 — Run tests

```powershell
python -m pytest -q
```

You want all tests to pass.

## Phase 6 — SAM2 comparison

The primary assignment methods remain classical OpenCV. SAM2 is only used for comparison.

Install the optional SAM2 comparison packages:

```powershell
pip install -r requirements_sam2.txt
```

Then compare the RGB result:

```powershell
python src\sam2_compare.py --image data\rgb\rgb_human.jpg --bbox 300 80 720 930 --classical-mask outputs\rgb\classical_mask.png --output-mask outputs\rgb\sam2_mask.png
```

Compare the thermal result:

```powershell
python src\sam2_compare.py --image data\thermal\thermal_human.jpg --bbox 190 150 450 340 --classical-mask outputs\thermal\classical_mask.png --output-mask outputs\thermal\sam2_mask.png
```

The script prints IoU and Dice between the classical mask and SAM2 mask.

## Important SAM2 note

The official SAM2 repository currently recommends Python 3.10+ and PyTorch 2.5.1+ and strongly recommends WSL on Windows. Hugging Face Transformers also supports SAM2 with box prompts. If the local SAM2 comparison becomes difficult on Windows, use WSL or follow the Transformers route in `requirements_sam2.txt`.

## Phase 7 — Web application

Run:

```powershell
streamlit run app\app.py
```

The app shows saved outputs first and also lets you upload a new image and run the classical method manually.

## Phase 8 — Report

Fill the placeholders in `report/report.md` with your actual images and SAM2 comparison numbers.

## Phase 9 — GitHub

The assignment requires a separate GitHub repository for Module 4. Initialize Git in the project root, commit the project, and push it to a new public repository.

## Phase 10 — Public course portal

After the Module 4 app is public, add its URL to your existing CSc 8830 Computer Vision portal so your professor can access all modules from one link.
