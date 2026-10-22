import os
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using compute device: {device}")

class FruitDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        return img, label, path

def compute_class_weights(dataset, num_classes):
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    total = len(dataset)
    # Balanced weighting: total / (num_classes * count)
    weights = total / (num_classes * counts + 1e-6)
    return torch.tensor(weights, dtype=torch.float32)

def train_and_evaluate_dl():
    base_dir = r"c:\Users\rravi\Downloads\Orange Dataset"
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    output_dir = r"c:\Users\rravi\Downloads\new\outputs"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. Datasets
    full_train_dataset = FruitDataset(train_dir, transform=train_transform)
    test_dataset = FruitDataset(test_dir, transform=test_transform)
    num_classes = len(full_train_dataset.classes)
    class_names = full_train_dataset.classes
    print(f"Detected {num_classes} classes: {class_names}")

    # Split train into train (85%) and val (15%)
    val_size = int(0.15 * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    torch.manual_seed(42)
    train_subset, val_subset = random_split(full_train_dataset, [train_size, val_size])

    # For validation, we use test_transform (no data augmentation)
    class SubsetEnabledTransformDataset(Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, idx):
            orig_idx = self.subset.indices[idx]
            path, label = self.subset.dataset.samples[orig_idx]
            with Image.open(path) as img:
                img = img.convert("RGB")
                if self.transform:
                    img = self.transform(img)
            return img, label, path

    val_dataset = SubsetEnabledTransformDataset(val_subset, test_transform)

    batch_size = 16
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights for loss
    class_weights = compute_class_weights(full_train_dataset, num_classes).to(device)
    print(f"Computed Class Weights (to balance ~2:1 imbalance): {class_weights.cpu().numpy().round(3)}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 3. Model: MobileNetV3-Small (lightweight deep learning architecture)
    print("\n--- Initializing Lightweight Deep Architecture: MobileNetV3-Small ---")
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,} | Trainable: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

    # 4. Training loop
    epochs = 12
    best_val_f1 = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_acc": []}

    print(f"\nStarting training for {epochs} epochs...")
    start_time = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_subset)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average='macro')

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | Val Macro F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())

    total_train_time = time.time() - start_time
    print(f"\nTraining completed in {total_train_time:.1f}s. Best Val Macro F1: {best_val_f1:.4f}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, os.path.join(output_dir, "best_mobilenet_v3.pth"))

    # 5. Test Evaluation & Latency Benchmarking
    print("\n--- Evaluating MobileNetV3 on Independent Test Set ---")
    model.eval()
    test_preds = []
    test_targets = []
    test_probs = []
    inference_times = []

    with torch.no_grad():
        for images, labels, _ in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            t0 = time.perf_counter()
            outputs = model(images)
            t1 = time.perf_counter()
            inference_times.append((t1 - t0) / images.size(0))

            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)

            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())

    test_acc = accuracy_score(test_targets, test_preds)
    test_f1_macro = f1_score(test_targets, test_preds, average='macro')
    test_f1_weighted = f1_score(test_targets, test_preds, average='weighted')
    test_prec_macro = precision_score(test_targets, test_preds, average='macro')
    test_rec_macro = recall_score(test_targets, test_preds, average='macro')
    mean_latency_ms = float(np.mean(inference_times) * 1000)

    print(f"MobileNetV3 Test Accuracy: {test_acc * 100:.2f}%")
    print(f"MobileNetV3 Macro F1: {test_f1_macro:.4f}")
    print(f"MobileNetV3 Macro Precision: {test_prec_macro:.4f}")
    print(f"MobileNetV3 Macro Recall: {test_rec_macro:.4f}")
    print(f"Mean Inference Latency: {mean_latency_ms:.2f} ms/image ({1000.0/mean_latency_ms:.1f} FPS)")
    print("\nMobileNetV3 Classification Report:")
    print(classification_report(test_targets, test_preds, target_names=class_names))

    # Save results
    results = {
        "model_name": "MobileNetV3-Small",
        "total_params": total_params,
        "train_time_sec": total_train_time,
        "latency_ms": mean_latency_ms,
        "fps": 1000.0 / mean_latency_ms,
        "accuracy": float(test_acc),
        "macro_f1": float(test_f1_macro),
        "weighted_f1": float(test_f1_weighted),
        "macro_precision": float(test_prec_macro),
        "macro_recall": float(test_rec_macro),
        "conf_matrix": confusion_matrix(test_targets, test_preds).tolist(),
        "preds": [int(p) for p in test_preds],
        "y_true": [int(t) for t in test_targets],
        "classes": class_names,
        "history": history
    }

    with open(os.path.join(output_dir, "deeplearning_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Deep learning pipeline completed! Results saved to {output_dir}")

if __name__ == "__main__":
    train_and_evaluate_dl()
