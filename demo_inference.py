import os
import sys
import time
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms, models

from pipeline_classical import ClassicalFeatureExtractor

def run_demo(image_path=None):
    output_dir = r"c:\Users\rravi\Downloads\new\outputs"
    base_dir = r"c:\Users\rravi\Downloads\Orange Dataset"

    # Default image if none provided: pick one test sample
    if image_path is None or not os.path.exists(image_path):
        # Pick a sample from test set
        test_dir = os.path.join(base_dir, "test")
        sample_classes = ["fresh", "canker", "blackspot", "grenning"]
        for sc in sample_classes:
            cls_folder = os.path.join(test_dir, sc)
            if os.path.exists(cls_folder) and os.listdir(cls_folder):
                image_path = os.path.join(cls_folder, os.listdir(cls_folder)[0])
                break

    print("="*70)
    print("DEMO INFERENCE PIPELINE: END-TO-END COMPUTER VISION RUN")
    print(f"Target Image: {image_path}")
    print("="*70)

    # Load image
    with Image.open(image_path) as im:
        raw_img = im.convert("RGB")

    # Load mappings
    with open(os.path.join(output_dir, "class_mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    class_to_idx = mappings["class_to_idx"]
    idx_to_class = mappings["idx_to_class"]
    class_names = [idx_to_class[i] for i in range(len(class_to_idx))]

    # -------------------------------------------------------------
    # 1. Approach 1: Classical CV Pipeline (Feature Extractor + SVM/RF)
    # -------------------------------------------------------------
    print("\n[Approach 1: Classical Computer Vision Pipeline]")
    with open(os.path.join(output_dir, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(output_dir, "svm_model.pkl"), "rb") as f:
        svm_model = pickle.load(f)
    with open(os.path.join(output_dir, "rf_model.pkl"), "rb") as f:
        rf_model = pickle.load(f)

    extractor = ClassicalFeatureExtractor()

    t0 = time.perf_counter()
    raw_feats = extractor.extract(raw_img)
    scaled_feats = scaler.transform([raw_feats])
    t1 = time.perf_counter()

    svm_probs = svm_model.predict_proba(scaled_feats)[0]
    rf_probs = rf_model.predict_proba(scaled_feats)[0]
    t2 = time.perf_counter()

    classical_feat_latency = (t1 - t0) * 1000.0
    classical_infer_latency = (t2 - t1) * 1000.0
    classical_total_latency = (t2 - t0) * 1000.0

    svm_pred_idx = np.argmax(svm_probs)
    svm_pred_cls = idx_to_class[svm_pred_idx]
    rf_pred_idx = np.argmax(rf_probs)
    rf_pred_cls = idx_to_class[rf_pred_idx]

    print(f"  > Handcrafted Features: 53-dim vector extracted in {classical_feat_latency:.2f} ms")
    print(f"    - Lesion / Defect Spot Area Ratio: {raw_feats[-2]*100:.2f}%")
    print(f"    - Mean Gradient Energy: {raw_feats[-6]:.2f}")
    print(f"  > SVM (RBF) Prediction: {svm_pred_cls.upper()} (Confidence: {svm_probs[svm_pred_idx]*100:.1f}%)")
    print(f"  > RF Prediction:       {rf_pred_cls.upper()} (Confidence: {rf_probs[rf_pred_idx]*100:.1f}%)")
    print(f"  > Total Classical Latency: {classical_total_latency:.2f} ms (Inference only: {classical_infer_latency:.2f} ms)")

    # -------------------------------------------------------------
    # 2. Approach 2: Lightweight Deep Learning (MobileNetV3)
    # -------------------------------------------------------------
    print("\n[Approach 2: Lightweight Deep Learning (MobileNetV3-Small)]")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    dl_model = models.mobilenet_v3_small()
    in_features = dl_model.classifier[3].in_features
    dl_model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(class_names))
    )
    weights_path = os.path.join(output_dir, "best_mobilenet_v3.pth")
    if os.path.exists(weights_path):
        dl_model.load_state_dict(torch.load(weights_path, map_location=device))
        dl_model.to(device)
        dl_model.eval()

        dl_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        t3 = time.perf_counter()
        input_tensor = dl_transform(raw_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = dl_model(input_tensor)
            probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
        t4 = time.perf_counter()
        dl_latency = (t4 - t3) * 1000.0

        dl_pred_idx = np.argmax(probs)
        dl_pred_cls = class_names[dl_pred_idx]
        print(f"  > MobileNetV3 Prediction: {dl_pred_cls.upper()} (Confidence: {probs[dl_pred_idx]*100:.1f}%)")
        print(f"  > MobileNetV3 Latency: {dl_latency:.2f} ms ({1000.0/dl_latency:.1f} FPS)")
    else:
        probs = None
        dl_pred_cls = "N/A"
        dl_latency = 0.0
        print("  > Deep Learning model weights not yet found.")

    # -------------------------------------------------------------
    # 3. Save Visual Inspection Demo Figure
    # -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # A: Raw Image
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Input Sample\nFile: {os.path.basename(image_path)}", fontsize=11, fontweight='bold')
    axes[0].axis('off')

    # B: Classical Defect / Color Mask Preview
    rgb_arr = np.array(raw_img.resize((256, 256)), dtype=np.float32)
    gray = 0.2989 * rgb_arr[..., 0] + 0.5870 * rgb_arr[..., 1] + 0.1140 * rgb_arr[..., 2]
    gx = np.zeros_like(gray)
    gx[:, 1:-1] = np.abs(gray[:, 2:] - gray[:, :-2]) / 2.0
    axes[1].imshow(gx, cmap='magma')
    axes[1].set_title(f"Classical Gradient Energy & Edges\n(Defect Area: {raw_feats[-2]*100:.1f}%)", fontsize=11, fontweight='bold')
    axes[1].axis('off')

    # C: Model Confidence Comparison Bar Chart
    y_pos = np.arange(len(class_names))
    height = 0.35
    axes[2].barh(y_pos - height/2, svm_probs * 100, height, label='Classical SVM (RBF)', color='#2ecc71')
    if probs is not None:
        axes[2].barh(y_pos + height/2, probs * 100, height, label='MobileNetV3', color='#9b59b6')
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(class_names, fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Prediction Confidence (%)', fontsize=11)
    axes[2].set_title(f"Class Probabilities\nClassical: {svm_pred_cls} | DL: {dl_pred_cls}", fontsize=11, fontweight='bold')
    axes[2].set_xlim(0, 105)
    axes[2].legend(loc='lower right')
    axes[2].grid(axis='x', linestyle='--', alpha=0.5)

    plt.tight_layout()
    demo_plot_path = os.path.join(output_dir, "example_run_inference.png")
    plt.savefig(demo_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nVisual inference demo plot generated: {demo_plot_path}")
    print("="*70)

if __name__ == "__main__":
    test_img = sys.argv[1] if len(sys.argv) > 1 else None
    run_demo(test_img)
