# GreenPod Labs Computer Vision Pipeline: Fruit Ripening & Defect Classification

This repository contains the complete, production-ready computer vision pipeline developed for **GreenPod Labs Data Science Assignment-2**.

---

## 📌 Dataset Note
The assignment prompt mentions an *"Avocado ripening stage"* classification task, but the dataset provided at the Google Drive URL (`https://drive.google.com/file/d/1NMJ0FOxcWH1mwFFeERuY6fgEY-MVgeET/view`) is **`Orange Dataset.zip`**, which contains 1,090 citrus fruit images categorized into 4 classes:
- `fresh` (Healthy / Prime stage)
- `grenning` (Citrus Greening disease / Unripe physiological disorder)
- `canker` (Citrus bacterial canker lesions)
- `blackspot` (Citrus black spot fungal lesions)

Our computer vision architecture is **completely generalizable** and directly applicable to multi-stage fruit ripening (e.g. Avocado firm/breaking/ripe/overripe) as well as fruit surface defect sorting.

---

## 🏗️ Implemented Approaches

1. **Approach 1: Classical Computer Vision & Handcrafted Machine Learning Pipeline**
   - **Feature Extraction (53 dimensions):**
     - Multi-space color statistical moments (Mean, Variance, Skewness across RGB, HSV, and CIE L\*a\*b\* spaces).
     - Quantized 2D color histograms (Hue: 12 bins, Saturation: 8 bins).
     - Surface texture and gradient energy via Sobel gradient magnitude (Mean, Std, Max, 90th percentile).
     - Defect lesion segmentation (luminance deficit ratio and mean lesion luminance).
   - **Classifiers:** Support Vector Machine (RBF kernel) and Balanced Random Forest.
   - **Performance:** **97.98% Test Accuracy, 0.9772 Macro F1** (SVM RBF).

2. **Approach 2: Lightweight Deep Learning Architecture (MobileNetV3-Small)**
   - **Backbone:** Pretrained MobileNetV3-Small (~1.52M parameters, ~5.8 MB).
   - **Class-Imbalance Mitigation:** Weighted Cross-Entropy Loss with inverse class frequency weights `[1.346, 1.384, 0.882, 0.714]`.
   - **Data Augmentation:** Random Horizontal/Vertical flips, rotation (±20°), color jittering, and ImageNet standardization.
   - **Performance:** **95.96% Test Accuracy, 0.9570 Macro F1**, **7.88 ms/image latency (127 FPS on CPU)**.

---

## 🚀 Quickstart & Reproduction

### 1. Requirements
Ensure Python 3.10+ is installed with the required libraries:
```bash
pip install torch torchvision scikit-learn opencv-python matplotlib seaborn scipy pandas numpy pillow
```

### 2. Run Exploratory Data Analysis (EDA)
```bash
python eda_dataset.py
```

### 3. Run Classical CV Pipeline
```bash
python pipeline_classical.py
```
Outputs model weights (`outputs/svm_model.pkl`, `outputs/rf_model.pkl`, `outputs/scaler.pkl`) and `outputs/classical_results.json`.

### 4. Run Deep Learning Pipeline
```bash
python pipeline_deeplearning.py
```
Trains MobileNetV3-Small for 12 epochs with early checkpointing, saving `outputs/best_mobilenet_v3.pth` and `outputs/deeplearning_results.json`.

### 5. Generate Comparison Plots & Visualizations
```bash
python evaluate_and_compare.py
```
Saves high-resolution plots to `outputs/visualizations/`:
- `confusion_matrices_comparison.png`
- `metrics_comparison.png`
- `per_class_f1_comparison.png`
- `mobilenet_training_curves.png`

### 6. Run End-to-End Demo Inference
```bash
# Test on default sample
python demo_inference.py

# Test on any specific image
python demo_inference.py "c:\Users\rravi\Downloads\Orange Dataset\test\canker\c (1).jpg"
```
Produces an inspection chart `outputs/example_run_inference.png` with input image, gradient energy map, and probability bar chart.
