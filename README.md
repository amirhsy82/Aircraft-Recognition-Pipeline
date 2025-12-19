# Military Aircraft Recognition Pipeline

An end-to-end computer vision pipeline for detecting and classifying military aircraft using the **Military Aircraft Recognition Dataset**. This project features a custom data engineering suite, coordinate-aware image preprocessing, and automated model optimization via KerasTuner.

## Project Overview
The challenge of this dataset lies in the varying aspect ratios of aircraft and the presence of multiple objects per scene. This pipeline addresses these challenges by:
1.  **Extracting** individual objects using Pascal VOC XML annotations.
2.  **Standardizing** inputs using aspect-ratio-preserving padding (Letterboxing).
3.  **Optimizing** CNN architecture automatically using the Hyperband algorithm.

---

##  Dataset Structure
The project is designed to work with the Pascal VOC format. The pipeline automatically downloads and organizes data from Kaggle:
* **JPEGImages**: Raw aircraft photos.
* **Annotations**: XML files containing bounding box coordinates ($x_{min}, y_{min}, x_{max}, y_{max}$).
* **ImageSets**: Text files defining training and testing splits.



---

## Data Engineering & Preprocessing

### 1. Coordinate-Aware Transformation
Standard resizing distorts the geometric features of aircraft (making them look squashed). Our pipeline implements a **Square Padding** strategy:
* Identifies the longest dimension of the aircraft crop.
* Pads the shorter dimension with black pixels.
* Resizes to a uniform $64 \times 64$ resolution.
* **Synchronized Labels:** Bounding box coordinates are mathematically re-mapped to ensure the ground truth stays aligned with the transformed image.



### 2. The Preprocessing Factory
The `preprocessing()` function handles the terminal stage of the data line:
* **ROI Extraction:** Cuts aircraft out of the original large-scale scenes.
* **Normalization:** Scales pixel intensities from $[0, 255]$ to $[0.0, 1.0]$ for faster convergence.
* **Label Encoding:** Applies `OneHotEncoder` to convert categorical labels into machine-readable unit vectors.



---

##  Automated Model Optimization
Instead of manual architecture design, we utilize **KerasTuner's Hyperband** algorithm to discover the optimal CNN configuration.

### Search Space:
* **Convolutional Layers:** 1 to 3 blocks.
* **Filter Density:** 32, 64, 96, or 128 filters per layer.
* **Dropout Rate:** 0% to 50% for regularization.
* **Learning Rate:** Optimized between $10^{-2}$ and $10^{-4}$.



---

## Evaluation & Results

### Model Architecture
The final architecture is a specialized CNN designed for $64 \times 64$ RGB inputs, featuring Max-Pooling for spatial reduction and Dropout for generalization.

### Performance Analysis
We evaluate the model using Accuracy and Loss curves to ensure the training process is stable and free from significant overfitting.



```text
Final Test Results:
- Test Loss: 0.XXXX
- Test Accuracy: XX.XX%
