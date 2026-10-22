import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def generate_comparisons():
    output_dir = r"c:\Users\rravi\Downloads\new\outputs"
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    classical_file = os.path.join(output_dir, "classical_results.json")
    dl_file = os.path.join(output_dir, "deeplearning_results.json")

    if not os.path.exists(classical_file) or not os.path.exists(dl_file):
        print(f"Waiting for result files to exist: classical={os.path.exists(classical_file)}, dl={os.path.exists(dl_file)}")
        return

    with open(classical_file, "r") as f:
        classical_res = json.load(f)
    with open(dl_file, "r") as f:
        dl_res = json.load(f)

    classes = classical_res["classes"]
    y_true = np.array(classical_res["y_true"])

    # 1. Confusion Matrix Comparison
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Random Forest CM
    cm_rf = np.array(classical_res["rf"]["conf_matrix"])
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f"Random Forest (Macro F1: {classical_res['rf']['macro_f1']:.3f})", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # SVM CM
    cm_svm = np.array(classical_res["svm"]["conf_matrix"])
    sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens", xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title(f"Support Vector Machine (Macro F1: {classical_res['svm']['macro_f1']:.3f})", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Predicted Label")
    axes[1].set_ylabel("True Label")

    # MobileNetV3 CM
    cm_dl = np.array(dl_res["conf_matrix"])
    sns.heatmap(cm_dl, annot=True, fmt="d", cmap="Purples", xticklabels=classes, yticklabels=classes, ax=axes[2])
    axes[2].set_title(f"MobileNetV3-Small (Macro F1: {dl_res['macro_f1']:.3f})", fontsize=13, fontweight='bold')
    axes[2].set_xlabel("Predicted Label")
    axes[2].set_ylabel("True Label")

    plt.suptitle("Confusion Matrix Comparison on Independent Test Set (N=99)", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    cm_path = os.path.join(vis_dir, "confusion_matrices_comparison.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {cm_path}")

    # 2. Overall Metrics Bar Chart
    metrics = ["Accuracy", "Macro F1", "Weighted F1", "Macro Precision", "Macro Recall"]
    rf_vals = [
        classical_res["rf"]["accuracy"],
        classical_res["rf"]["macro_f1"],
        classical_res["rf"]["weighted_f1"],
        classical_res["rf"]["macro_precision"],
        classical_res["rf"]["macro_recall"]
    ]
    svm_vals = [
        classical_res["svm"]["accuracy"],
        classical_res["svm"]["macro_f1"],
        classical_res["svm"]["weighted_f1"],
        classical_res["svm"]["macro_precision"],
        classical_res["svm"]["macro_recall"]
    ]
    dl_vals = [
        dl_res["accuracy"],
        dl_res["macro_f1"],
        dl_res["weighted_f1"],
        dl_res["macro_precision"],
        dl_res["macro_recall"]
    ]

    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width, [v * 100 for v in rf_vals], width, label="Random Forest (Classical CV)", color="#3498db")
    rects2 = ax.bar(x, [v * 100 for v in svm_vals], width, label="SVM RBF (Classical CV)", color="#2ecc71")
    rects3 = ax.bar(x + width, [v * 100 for v in dl_vals], width, label="MobileNetV3-Small (Deep Learning)", color="#9b59b6")

    ax.set_ylabel("Percentage (%)", fontsize=12)
    ax.set_title("Overall Model Performance Metrics Comparison", fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    metrics_path = os.path.join(vis_dir, "metrics_comparison.png")
    plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {metrics_path}")

    # 3. Per-Class F1 Score Breakdown
    from sklearn.metrics import f1_score
    rf_preds = np.array(classical_res["rf"]["preds"])
    svm_preds = np.array(classical_res["svm"]["preds"])
    dl_preds = np.array(dl_res["preds"])

    f1_rf_per_class = f1_score(y_true, rf_preds, average=None)
    f1_svm_per_class = f1_score(y_true, svm_preds, average=None)
    f1_dl_per_class = f1_score(y_true, dl_preds, average=None)

    fig, ax = plt.subplots(figsize=(10, 5))
    x_cls = np.arange(len(classes))
    ax.bar(x_cls - width, [v * 100 for v in f1_rf_per_class], width, label="Random Forest", color="#3498db")
    ax.bar(x_cls, [v * 100 for v in f1_svm_per_class], width, label="SVM RBF", color="#2ecc71")
    ax.bar(x_cls + width, [v * 100 for v in f1_dl_per_class], width, label="MobileNetV3-Small", color="#9b59b6")

    ax.set_ylabel("F1 Score (%)", fontsize=12)
    ax.set_title("Per-Class F1-Score Breakdown (Robustness across Defect Types)", fontsize=14, fontweight='bold')
    ax.set_xticks(x_cls)
    ax.set_xticklabels(classes, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right")
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for i in range(len(classes)):
        ax.annotate(f"{f1_rf_per_class[i]*100:.1f}%", (x_cls[i]-width, f1_rf_per_class[i]*100 + 2), ha='center', fontsize=9)
        ax.annotate(f"{f1_svm_per_class[i]*100:.1f}%", (x_cls[i], f1_svm_per_class[i]*100 + 2), ha='center', fontsize=9)
        ax.annotate(f"{f1_dl_per_class[i]*100:.1f}%", (x_cls[i]+width, f1_dl_per_class[i]*100 + 2), ha='center', fontsize=9)

    plt.tight_layout()
    per_class_path = os.path.join(vis_dir, "per_class_f1_comparison.png")
    plt.savefig(per_class_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {per_class_path}")

    # 4. Deep Learning Training History Curve
    if "history" in dl_res:
        hist = dl_res["history"]
        epochs = list(range(1, len(hist["train_loss"]) + 1))
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color = 'tab:red'
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', color=color, fontsize=12, fontweight='bold')
        ax1.plot(epochs, hist["train_loss"], 'o--', color='tab:red', label='Train Loss')
        ax1.plot(epochs, hist["val_loss"], 's-', color='tab:orange', label='Val Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle=':', alpha=0.6)

        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Validation F1 (%)', color=color, fontsize=12, fontweight='bold')
        ax2.plot(epochs, [v * 100 for v in hist["val_f1"]], '^--', color='tab:blue', label='Val Macro F1')
        ax2.plot(epochs, [v * 100 for v in hist["val_acc"]], 'd-', color='tab:purple', label='Val Accuracy')
        ax2.tick_params(axis='y', labelcolor=color)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
        plt.title("MobileNetV3 Convergence & Validation Trajectory", fontsize=14, fontweight='bold')

        plt.tight_layout()
        train_hist_path = os.path.join(vis_dir, "mobilenet_training_curves.png")
        plt.savefig(train_hist_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {train_hist_path}")

    print("\nAll comparison visualizations successfully generated!")

if __name__ == "__main__":
    generate_comparisons()
