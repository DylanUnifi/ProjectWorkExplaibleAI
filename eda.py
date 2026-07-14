import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score
import umap
import wandb

# Import loaders
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import get_clevr_hans_loaders
from dataset import get_mnmath_loaders

def parse_args():
    parser = argparse.ArgumentParser(description="Exploratory Data Analysis (EDA) based on Wasserman & Bishop.")
    parser.add_argument("--max_samples", type=int, default=1000, help="Max samples to analyze (to save memory/time).")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--dataset", type=str, required=True, choices=["clevr_hans3", "mnmath"], help="Dataset to analyze")
    return parser.parse_args()


def plot_to_wandb(fig, tag):
    wandb.log({tag: wandb.Image(fig)})
    plt.close(fig)


def analyze_class_distribution(all_labels, dataset_name):
    print("  [1/7] Analyzing Class Distribution (ISLR)...")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.countplot(x=all_labels, ax=ax, palette="viridis")
    ax.set_title(f"Class Distribution - {dataset_name}")
    ax.set_xlabel("Class Label")
    ax.set_ylabel("Count")
    
    counts = np.bincount(all_labels)
    total = len(all_labels)
    imbalance_ratio = np.max(counts) / (np.min(counts) + 1e-5)
    wandb.log({
        f"{dataset_name}_max_class_imbalance_ratio": imbalance_ratio,
        f"{dataset_name}_total_samples": total
    })
    
    plot_to_wandb(fig, f"{dataset_name}/Class_Distribution")


def analyze_pixel_intensities(all_images, dataset_name):
    print("  [2/7] Analyzing Pixel Intensities & eCDF (Wasserman)...")
    flat_pixels = all_images.flatten()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Histogram
    sns.histplot(flat_pixels, bins=50, kde=False, color="blue", stat="density", ax=axes[0])
    axes[0].set_title(f"Pixel Intensity Histogram - {dataset_name}")
    axes[0].set_xlabel("Normalized Pixel Value")
    
    # 2. Empirical CDF
    sns.ecdfplot(flat_pixels, color="red", ax=axes[1])
    axes[1].set_title(f"Empirical CDF - {dataset_name}")
    axes[1].set_xlabel("Normalized Pixel Value")
    axes[1].set_ylabel("CDF")
    
    plot_to_wandb(fig, f"{dataset_name}/Pixel_Intensities_and_eCDF")
    
    if all_images.shape[1] == 3:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['red', 'green', 'blue']
        for c in range(3):
            sns.kdeplot(all_images[:, c, :, :].flatten(), color=colors[c], label=f"Channel {c}", ax=ax)
        ax.set_title(f"Per-Channel Pixel Intensity (KDE) - {dataset_name}")
        ax.legend()
        plot_to_wandb(fig, f"{dataset_name}/Per_Channel_Intensities")


def normalize_img_for_display(img):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    if img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0)) # (H, W, C)
    else:
        img = img[0] # (H, W)
    return img


def analyze_mean_and_variance_images(all_images, all_labels, dataset_name):
    print("  [3/7] Computing Mean and Variance Images per Class (Bishop)...")
    classes = np.unique(all_labels)
    n_classes = len(classes)
    
    fig_mean, axes_mean = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    fig_var, axes_var = plt.subplots(1, n_classes, figsize=(4 * n_classes, 4))
    if n_classes == 1:
        axes_mean = [axes_mean]
        axes_var = [axes_var]
        
    for idx, c in enumerate(classes):
        class_imgs = all_images[all_labels == c]
        
        # Mean
        mean_img = np.mean(class_imgs, axis=0)
        ax_m = axes_mean[idx]
        if len(mean_img.shape) == 2 or mean_img.shape[0] == 1:
            ax_m.imshow(normalize_img_for_display(mean_img), cmap="gray")
        else:
            ax_m.imshow(normalize_img_for_display(mean_img))
        ax_m.set_title(f"Class {c} Mean Image")
        ax_m.axis("off")
        
        # Variance
        var_img = np.var(class_imgs, axis=0)
        ax_v = axes_var[idx]
        if len(var_img.shape) == 2 or var_img.shape[0] == 1:
            ax_v.imshow(normalize_img_for_display(var_img), cmap="magma")
        else:
            # For RGB, average variance across channels for a heatmap
            var_img_gray = np.mean(var_img, axis=0)
            ax_v.imshow(var_img_gray, cmap="magma")
        ax_v.set_title(f"Class {c} Variance")
        ax_v.axis("off")
        
    plot_to_wandb(fig_mean, f"{dataset_name}/Mean_Images")
    plot_to_wandb(fig_var, f"{dataset_name}/Variance_Images")


def analyze_manifold_and_complexity(all_images, all_labels, dataset_name):
    print("  [4/7] Computing PCA, t-SNE, UMAP and Entanglement Metrics...")
    N = all_images.shape[0]
    X_flat = all_images.reshape(N, -1)
    
    # Complexity Metric: Silhouette Score
    if len(np.unique(all_labels)) > 1:
        # Sample for speed if dataset is large
        sample_idx = np.random.choice(N, min(N, 500), replace=False)
        sil_score = silhouette_score(X_flat[sample_idx], all_labels[sample_idx])
        wandb.log({f"{dataset_name}_silhouette_score": sil_score})
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_flat)
    
    # Save Eigenimages (top 4 components)
    pca_n = PCA(n_components=4)
    pca_n.fit(X_flat)
    fig_eigen, axes_eigen = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        eigen_img = pca_n.components_[i].reshape(all_images.shape[1:])
        if eigen_img.shape[0] == 1:
            axes_eigen[i].imshow(eigen_img[0], cmap="gray")
        else:
            # RGB eigenimage normalization for display
            eigen_img_norm = (eigen_img - eigen_img.min()) / (eigen_img.max() - eigen_img.min() + 1e-8)
            axes_eigen[i].imshow(np.transpose(eigen_img_norm, (1, 2, 0)))
        axes_eigen[i].set_title(f"PC{i+1} (Var: {pca_n.explained_variance_ratio_[i]:.2%})")
        axes_eigen[i].axis("off")
    plot_to_wandb(fig_eigen, f"{dataset_name}/Eigenimages")
    
    if X_flat.shape[1] > 50:
        pca_50 = PCA(n_components=min(50, N))
        X_flat_50 = pca_50.fit_transform(X_flat)
    else:
        X_flat_50 = X_flat
        
    tsne = TSNE(n_components=2, perplexity=min(30, N-1), random_state=42)
    X_tsne = tsne.fit_transform(X_flat_50)
    
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = umap_reducer.fit_transform(X_flat_50)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=all_labels, palette="viridis", ax=axes[0], legend="full", alpha=0.7)
    axes[0].set_title(f"PCA (Expl. Var: {sum(pca.explained_variance_ratio_):.2%})")
    
    sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=all_labels, palette="viridis", ax=axes[1], legend="full", alpha=0.7)
    axes[1].set_title("t-SNE Manifold")
    
    sns.scatterplot(x=X_umap[:, 0], y=X_umap[:, 1], hue=all_labels, palette="viridis", ax=axes[2], legend="full", alpha=0.7)
    axes[2].set_title("UMAP Manifold")
    
    plot_to_wandb(fig, f"{dataset_name}/Manifold_Comparisons")


def analyze_outliers(all_images, dataset_name):
    print("  [5/7] Detecting Outliers / Anomalies (Goodfellow)...")
    N = all_images.shape[0]
    X_flat = all_images.reshape(N, -1)
    
    # Use Isolation Forest on PCA-reduced data for speed
    pca_50 = PCA(n_components=min(50, N))
    X_reduced = pca_50.fit_transform(X_flat)
    
    iso = IsolationForest(contamination=0.05, random_state=42)
    iso.fit(X_reduced)
    scores = iso.decision_function(X_reduced) # Lower is more abnormal
    
    # Get 5 most abnormal images
    abnormal_idx = np.argsort(scores)[:5]
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i, idx in enumerate(abnormal_idx):
        img = normalize_img_for_display(all_images[idx])
        if len(img.shape) == 2:
            axes[i].imshow(img, cmap="gray")
        else:
            axes[i].imshow(img)
        axes[i].set_title(f"Anomaly Score: {scores[idx]:.2f}")
        axes[i].axis("off")
        
    plot_to_wandb(fig, f"{dataset_name}/Top_Outliers")


def analyze_confounders(all_confounded, dataset_name):
    if len(all_confounded) == 0 or all_confounded[0] is None:
        return
    print("  [6/7] Analyzing Confounders (Dataset bias)...")
    
    confounded_counts = np.bincount(np.array(all_confounded, dtype=int))
    labels = ["Unconfounded", "Confounded"]
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(confounded_counts, labels=labels[:len(confounded_counts)], autopct='%1.1f%%', colors=["#2ecc71", "#e74c3c"])
    ax.set_title(f"Confounder Distribution - {dataset_name}")
    plot_to_wandb(fig, f"{dataset_name}/Confounder_Distribution")


def run_eda(dataset_name, loader):
    print(f"\n{'='*50}\nStarting EDA for {dataset_name}\n{'='*50}")
    
    all_images = []
    all_labels = []
    all_confounded = []
    
    print("Loading data into memory...")
    for batch in tqdm(loader, desc="Reading batches"):
        all_images.append(batch["image"].numpy())
        all_labels.append(batch["label"].numpy())
        if "confounded" in batch:
            all_confounded.append(batch["confounded"].numpy())
            
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    if all_confounded:
        all_confounded = np.concatenate(all_confounded, axis=0)
        
    print(f"Loaded {len(all_images)} samples. Shape: {all_images.shape}")
    
    analyze_class_distribution(all_labels, dataset_name)
    analyze_pixel_intensities(all_images, dataset_name)
    analyze_mean_and_variance_images(all_images, all_labels, dataset_name)
    analyze_manifold_and_complexity(all_images, all_labels, dataset_name)
    analyze_outliers(all_images, dataset_name)
    analyze_confounders(all_confounded, dataset_name)


def main():
    args = parse_args()
    
    wandb.init(
        project="XAI_Comparative_Study",
        name=f"EDA_{args.dataset}",
        job_type="Exploratory_Data_Analysis",
        config=vars(args)
    )
    
    if args.dataset == "clevr_hans3":
        train_loader, _, _ = get_clevr_hans_loaders(
            root_dir="./CLEVR-Hans3",
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        if train_loader is not None:
            run_eda("CLEVR-Hans3", train_loader)
            
    elif args.dataset == "mnmath":
        train_loader, _, _ = get_mnmath_loaders(
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        if train_loader is not None:
            run_eda("MNMath", train_loader)
            
    wandb.finish()
    print("\nEDA completed. All plots have been logged to Weights & Biases!")


if __name__ == "__main__":
    main()
