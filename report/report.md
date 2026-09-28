# CSc 8830 Computer Vision — Module 4
## Classical Human Boundary Segmentation and SAM2 Comparison

**Student:** Olumide Adebisi  
**GitHub:** https://github.com/Olumide1996/CSC8830_Module4

## 1. Objective

This project finds the boundary of a person in an RGB image and in a thermal image using classical OpenCV image-processing methods. The assignment does not allow machine-learning or deep-learning methods for the main implementation. I then compare the classical boundary with SAM2 as a reference method.

## 2. RGB Segmentation

[Insert RGB input image]

[Insert RGB classical mask/overlay]

The RGB pipeline uses an approximate bounding box supplied by the user, color contrast in the Lab color space, edge detection, morphology, connected-component cleanup, and contour extraction. No learned model is used.

## 3. Thermal Segmentation

[Insert thermal input image]

[Insert thermal classical mask/overlay]

The thermal pipeline uses the grayscale intensity of the thermal display image. A high-intensity threshold is followed by morphology and connected-component selection inside the supplied person box. No learned model is used.

## 4. SAM2 Comparison

[Insert SAM2 comparison images]

SAM2 was used only as a reference comparison. The same bounding box was given to SAM2 so that the two methods received the same basic object-location information. SAM2 supports box prompts for image segmentation. [1]

| Modality | Classical vs. SAM2 IoU | Dice |
|---|---:|---:|
| RGB | [fill] | [fill] |
| Thermal | [fill] | [fill] |

These values measure agreement between the two masks. They should not be described as ground-truth accuracy because SAM2 itself is not a manually verified ground truth.

## 5. Fourier-Domain Theory

For an image \(f(x,y)\), spatial differentiation becomes multiplication in the Fourier domain:

\[
\mathcal{F}\left\{\frac{\partial f}{\partial x}\right\}=j2\pi uF(u,v),
\qquad
\mathcal{F}\left\{\frac{\partial f}{\partial y}\right\}=j2\pi vF(u,v).
\]

Edges correspond to rapid spatial changes, so their energy is concentrated at higher spatial frequencies. A high-pass filter can therefore emphasize edges before a region is separated or a boundary is extracted.

For the Laplacian,

\[
\mathcal{F}\{\nabla^2f\}=-4\pi^2(u^2+v^2)F(u,v).
\]

This shows directly why the Laplacian emphasizes high-frequency components associated with edges.

Segmentation itself is a nonlinear decision that produces a region mask \(M(x,y)\). Fourier filtering can be used as a preprocessing step to smooth noise or strengthen boundaries, after which thresholding, connected components, or contour extraction can produce the final mask.

## 6. Results and Discussion

The classical RGB and thermal methods use only traditional image-processing operations. Their results depend on the input image, the selected bounding box, lighting or thermal contrast, and the quality of the boundaries in the source image.

The SAM2 comparison shows how closely the classical boundary follows a modern promptable segmentation reference. The comparison should be interpreted as method agreement rather than an accuracy score.

## 7. Web Application

The Streamlit application allows the user to upload an RGB or thermal image, define the approximate human region, run the classical segmentation method, and view the mask, contour, and Fourier edge map. Saved experiment outputs can also be displayed directly in the web interface.

## 8. Conclusion

The project demonstrates that human boundaries can be extracted without a learned model by combining basic color or thermal-intensity separation with edges, morphology, connected components, and contours. Fourier-domain filtering gives another way to emphasize the image changes that form object boundaries. SAM2 provides a useful reference for comparing the resulting masks, while the classical OpenCV pipeline satisfies the main implementation restriction.

### Reference

[1] Meta AI, “SAM 2: Segment Anything in Images and Videos,” official research and repository documentation. https://ai.meta.com/research/sam2/ and https://github.com/facebookresearch/sam2
