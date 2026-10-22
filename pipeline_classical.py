import os
import glob
import pickle
import time
import numpy as np
from PIL import Image
from scipy.stats import skew
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

class ClassicalFeatureExtractor:
    """
    Extracts handcrafted computer vision features:
    1. Multi-space Color Moments (RGB, HSV, LAB: mean, std, skewness)
    2. 2D Color Histograms (Hue & Saturation)
    3. Gradient & Edge Texture (Sobel gradient magnitude mean, std, energy)
    4. Lesion / Defect Spot Segmentation (dark spot area ratio, contour count)
    """
    def __init__(self, target_size=(256, 256)):
        self.target_size = target_size

    def rgb_to_hsv(self, img_arr):
        # Normalize to 0-1
        arr = img_arr.astype(np.float32) / 255.0
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        maxc = np.maximum(np.maximum(r, g), b)
        minc = np.minimum(np.minimum(r, g), b)
        v = maxc
        delta = maxc - minc
        s = np.zeros_like(v)
        nonzero_mask = maxc > 1e-6
        s[nonzero_mask] = delta[nonzero_mask] / maxc[nonzero_mask]

        # Hue
        h = np.zeros_like(v)
        rc = (maxc - r) / (delta + 1e-6)
        gc = (maxc - g) / (delta + 1e-6)
        bc = (maxc - b) / (delta + 1e-6)

        r_mask = (r == maxc) & nonzero_mask
        g_mask = (g == maxc) & nonzero_mask
        b_mask = (b == maxc) & nonzero_mask

        h[r_mask] = bc[r_mask] - gc[r_mask]
        h[g_mask] = 2.0 + rc[g_mask] - bc[g_mask]
        h[b_mask] = 4.0 + gc[b_mask] - rc[b_mask]
        h = (h / 6.0) % 1.0
        return np.stack([h, s, v], axis=-1)

    def rgb_to_lab(self, img_arr):
        # Standard conversion approximation
        arr = img_arr.astype(np.float32) / 255.0
        # sRGB to linear RGB
        mask = arr > 0.04045
        arr[mask] = ((arr[mask] + 0.055) / 1.055) ** 2.4
        arr[~mask] = arr[~mask] / 12.92
        # Observer. = 2°, Illuminant = D65
        X = arr[..., 0] * 0.4124564 + arr[..., 1] * 0.3575761 + arr[..., 2] * 0.1804375
        Y = arr[..., 0] * 0.2126729 + arr[..., 1] * 0.7151522 + arr[..., 2] * 0.0721750
        Z = arr[..., 0] * 0.0193339 + arr[..., 1] * 0.1191920 + arr[..., 2] * 0.9503041

        # Normalize for D65
        X /= 0.95047
        Y /= 1.00000
        Z /= 1.08883

        def f(t):
            return np.where(t > 0.008856, t ** (1/3), (7.787 * t) + (16/116))

        fx = f(X)
        fy = f(Y)
        fz = f(Z)

        L = (116.0 * fy) - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return np.stack([L, a, b], axis=-1)

    def extract(self, img_path_or_pil):
        if isinstance(img_path_or_pil, str):
            with Image.open(img_path_or_pil) as im:
                img = im.convert("RGB").resize(self.target_size, Image.Resampling.BILINEAR)
        else:
            img = img_path_or_pil.convert("RGB").resize(self.target_size, Image.Resampling.BILINEAR)

        rgb = np.array(img, dtype=np.float32)
        hsv = self.rgb_to_hsv(rgb)
        lab = self.rgb_to_lab(rgb)

        features = []

        # 1. Color moments (mean, std, skewness) for RGB, HSV, LAB
        for space in [rgb, hsv, lab]:
            for ch in range(3):
                channel = space[..., ch].flatten()
                mean_val = np.mean(channel)
                std_val = np.std(channel)
                skew_val = float(skew(channel)) if std_val > 1e-6 else 0.0
                features.extend([mean_val, std_val, skew_val])

        # 2. Color Histogram (Hue 12 bins, Saturation 8 bins)
        h_channel = hsv[..., 0].flatten()
        s_channel = hsv[..., 1].flatten()
        h_hist, _ = np.histogram(h_channel, bins=12, range=(0.0, 1.0), density=True)
        s_hist, _ = np.histogram(s_channel, bins=8, range=(0.0, 1.0), density=True)
        features.extend(h_hist)
        features.extend(s_hist)

        # 3. Texture / Gradient Energy via Sobel approximation
        gray = 0.2989 * rgb[..., 0] + 0.5870 * rgb[..., 1] + 0.1140 * rgb[..., 2]
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) / 2.0
        gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) / 2.0
        grad_mag = np.sqrt(gx**2 + gy**2)

        features.append(np.mean(grad_mag))
        features.append(np.std(grad_mag))
        features.append(np.max(grad_mag))
        features.append(np.percentile(grad_mag, 90))

        # 4. Defect spot / lesion ratio (dark spots in lightness and high saturation)
        L = lab[..., 0]
        # Defect criteria: significantly low luminance compared to fruit body
        fruit_mask = gray > 20.0  # exclude black background if any
        if np.sum(fruit_mask) > 100:
            median_L = np.median(L[fruit_mask])
            lesion_mask = (L < (median_L * 0.65)) & fruit_mask
            lesion_ratio = np.sum(lesion_mask) / np.sum(fruit_mask)
            lesion_mean_L = np.mean(L[lesion_mask]) if np.sum(lesion_mask) > 0 else 0.0
        else:
            lesion_ratio = 0.0
            lesion_mean_L = 0.0

        features.append(lesion_ratio)
        features.append(lesion_mean_L)

        return np.array(features, dtype=np.float32)

def load_dataset(base_dir, split, extractor):
    split_dir = os.path.join(base_dir, split)
    classes = sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    X = []
    y = []
    filepaths = []

    print(f"Loading and extracting classical features for '{split}' set...")
    start_t = time.time()
    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(cls_dir, fname)
                try:
                    feat = extractor.extract(fpath)
                    X.append(feat)
                    y.append(class_to_idx[cls])
                    filepaths.append(fpath)
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")

    elapsed = time.time() - start_t
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    print(f"Extracted {len(X)} samples, feature dim: {X.shape[1]} in {elapsed:.2f}s")
    return X, y, class_to_idx, filepaths

def train_and_evaluate_classical():
    data_dir = r"c:\Users\rravi\Downloads\Orange Dataset"
    output_dir = r"c:\Users\rravi\Downloads\new\outputs"
    os.makedirs(output_dir, exist_ok=True)

    extractor = ClassicalFeatureExtractor()

    # 1. Load train and test sets
    X_train, y_train, class_to_idx, train_paths = load_dataset(data_dir, "train", extractor)
    X_test, y_test, _, test_paths = load_dataset(data_dir, "test", extractor)
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 2. Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model A: Random Forest Classifier
    print("\n--- Training Classical Model A: Random Forest ---")
    rf = RandomForestClassifier(n_estimators=150, max_depth=12, class_weight='balanced', random_state=42)
    rf.fit(X_train_scaled, y_train)

    rf_train_preds = rf.predict(X_train_scaled)
    rf_test_preds = rf.predict(X_test_scaled)

    rf_acc = accuracy_score(y_test, rf_test_preds)
    rf_f1_macro = f1_score(y_test, rf_test_preds, average='macro')
    rf_f1_weighted = f1_score(y_test, rf_test_preds, average='weighted')
    rf_prec_macro = precision_score(y_test, rf_test_preds, average='macro')
    rf_rec_macro = recall_score(y_test, rf_test_preds, average='macro')

    print(f"Random Forest Test Accuracy: {rf_acc * 100:.2f}%")
    print(f"Random Forest Macro F1: {rf_f1_macro:.4f}")
    print(f"Random Forest Macro Precision: {rf_prec_macro:.4f}")
    print(f"Random Forest Macro Recall: {rf_rec_macro:.4f}")
    print("\nRandom Forest Classification Report:")
    target_names = [idx_to_class[i] for i in range(len(class_to_idx))]
    print(classification_report(y_test, rf_test_preds, target_names=target_names))

    # 4. Model B: Support Vector Machine (RBF)
    print("\n--- Training Classical Model B: Support Vector Machine (RBF) ---")
    svm = SVC(C=5.0, kernel='rbf', gamma='scale', probability=True, class_weight='balanced', random_state=42)
    svm.fit(X_train_scaled, y_train)

    svm_test_preds = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_test_preds)
    svm_f1_macro = f1_score(y_test, svm_test_preds, average='macro')
    svm_f1_weighted = f1_score(y_test, svm_test_preds, average='weighted')
    svm_prec_macro = precision_score(y_test, svm_test_preds, average='macro')
    svm_rec_macro = recall_score(y_test, svm_test_preds, average='macro')

    print(f"SVM Test Accuracy: {svm_acc * 100:.2f}%")
    print(f"SVM Macro F1: {svm_f1_macro:.4f}")
    print(f"SVM Macro Precision: {svm_prec_macro:.4f}")
    print(f"SVM Macro Recall: {svm_rec_macro:.4f}")
    print("\nSVM Classification Report:")
    print(classification_report(y_test, svm_test_preds, target_names=target_names))

    # Choose best classical model
    best_name = "RandomForest" if rf_f1_macro >= svm_f1_macro else "SVM"
    best_model = rf if best_name == "RandomForest" else svm

    # Save models and scaler
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(output_dir, "rf_model.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(output_dir, "svm_model.pkl"), "wb") as f:
        pickle.dump(svm, f)
    with open(os.path.join(output_dir, "class_mappings.pkl"), "wb") as f:
        pickle.dump({"class_to_idx": class_to_idx, "idx_to_class": idx_to_class}, f)

    results = {
        "rf": {
            "accuracy": float(rf_acc),
            "macro_f1": float(rf_f1_macro),
            "weighted_f1": float(rf_f1_weighted),
            "macro_precision": float(rf_prec_macro),
            "macro_recall": float(rf_rec_macro),
            "conf_matrix": confusion_matrix(y_test, rf_test_preds).tolist(),
            "preds": rf_test_preds.tolist()
        },
        "svm": {
            "accuracy": float(svm_acc),
            "macro_f1": float(svm_f1_macro),
            "weighted_f1": float(svm_f1_weighted),
            "macro_precision": float(svm_prec_macro),
            "macro_recall": float(svm_rec_macro),
            "conf_matrix": confusion_matrix(y_test, svm_test_preds).tolist(),
            "preds": svm_test_preds.tolist()
        },
        "y_true": y_test.tolist(),
        "classes": target_names
    }

    import json
    with open(os.path.join(output_dir, "classical_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nClassical pipeline completed successfully! Artifacts saved to {output_dir}")

if __name__ == "__main__":
    train_and_evaluate_classical()
