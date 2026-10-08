"""t-SNE visualisation of the fused 1024-d latent space."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def extract_latent_features(model, dataloader, device):
    """
    Hooks the penultimate layer g''_fuse (1024-d) from MGTFNet and extracts latent vectors.
    """
    model.eval()
    features = []
    labels_list = []
    
    # Store latent feature via forward hook
    latent_cache = {}
    def hook_fn(module, input, output):
        latent_cache['feat'] = output.detach().cpu()
        
    # Hook the fusion projection block output
    handle = model.fusion_projection.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            _ = model(images)  # Triggers hook
            features.append(latent_cache['feat'])
            labels_list.append(labels)

    handle.remove()
    features = torch.cat(features, dim=0).numpy()
    labels_list = torch.cat(labels_list, dim=0).numpy()
    return features, labels_list

def plot_tsne_latent_space(features, labels, class_names, save_path="tsne_feature_space.png"):
    """
    Generates a 2D t-SNE visualization of the 1024-d latent space (Figure 9 in paper).
    """
    print(f"\nComputing 2D t-SNE for {features.shape[0]} test samples...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    embedded = tsne.fit_transform(features)

    # Marker and palette configurations matching the paper's 10 classes
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*', '<', '>']
    palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    fig, ax = plt.subplots(figsize=(11, 9), dpi=300)
    for idx, class_name in enumerate(class_names):
        mask = (labels == idx)
        ax.scatter(
            embedded[mask, 0],
            embedded[mask, 1],
            c=palette[idx],
            marker=markers[idx],
            label=class_name,
            alpha=0.75,
            edgecolors='none',
            s=45
        )

    ax.set_title("t-SNE Visualization of MGTF-Net Latent Feature Space ($g''_{fuse}$ on MCFD-9K Test Set)", 
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=11, fontweight='bold')
    ax.set_ylabel("t-SNE Dimension 2", fontsize=11, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(
        title="Disease / Healthy Classes",
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        fontsize=9,
        title_fontsize=10,
        frameon=True
    )
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    print(f"t-SNE plot saved successfully as {save_path}")

