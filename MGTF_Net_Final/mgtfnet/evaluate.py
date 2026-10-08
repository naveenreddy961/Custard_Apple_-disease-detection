"""Metrics, confusion matrix, ROC and training curves."""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, cohen_kappa_score
from .config import Config

def evaluate_and_plot(model, test_loader, class_names, model_name="MGTFNet", weights_path=None):
    if weights_path:
        model.load_state_dict(torch.load(weights_path))
    model.eval()
    
    y_true = []
    y_pred = []
    y_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            y_probs.extend(probs.cpu().numpy())
            y_pred.extend(outputs.argmax(dim=1).cpu().numpy())
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Classification Report
    print(f"\n================ Classification Report: {model_name} ================")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    kappa = cohen_kappa_score(y_true, y_pred)
    print(f"Cohen's Kappa (κ): {kappa:.4f}")

    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f"{model_name}_confusion_matrix.png"), dpi=300)
    plt.show()

    # One-vs-Rest ROC Curve & Mean AUC
    plt.figure(figsize=(10, 8))
    aucs = []
    for i in range(len(class_names)):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_probs[:, i])
        class_auc = auc(fpr, tpr)
        aucs.append(class_auc)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {class_auc:.4f})")

    mean_auc = np.mean(aucs)
    plt.plot([0, 1], [0, 1], 'k--', label="Chance Level")
    plt.title(f"One-vs-Rest ROC Curves - {model_name} (mAUC = {mean_auc:.4f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f"{model_name}_roc_curves.png"), dpi=300)
    plt.show()

def plot_history(history, model_name="MGTFNet"):
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Validation Accuracy', linewidth=2, linestyle='--')
    plt.title(f"{model_name} - Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2, linestyle='--')
    plt.title(f"{model_name} - Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Focal Loss")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.OUTPUT_DIR, f"{model_name}_training_curves.png"), dpi=300)
    plt.show()
