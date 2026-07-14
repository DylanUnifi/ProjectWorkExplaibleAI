# explainability/advanced_visualization.py

"""
Advanced visualization for comparative XAI study.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec

def plot_comparison_grid(image, explanations_dict, save_path, ground_truth=None):
    """
    Plot comprehensive comparison grid of all explanations.
    
    Args:
        image: (C, H, W) original image
        explanations_dict: Dict[model_name][method_name] = attribution
        save_path: Path to save figure
        ground_truth: Optional (H, W) mask of important regions
    """
    
    # Flatten explanations
    all_explanations = []
    labels = []
    
    for model_name, methods in explanations_dict.items():
        for method_name, attr in methods.items():
            all_explanations.append(attr[0] if attr.dim() == 4 else attr)
            labels.append(f"{model_name}\n{method_name}")
    
    n_explanations = len(all_explanations)
    
    # Create grid
    n_cols = 5
    n_rows = (n_explanations + n_cols) // n_cols + 1  # +1 for original
    
    fig = plt.figure(figsize=(4*n_cols, 4*n_rows))
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    # Original image
    ax = fig.add_subplot(gs[0, :2])
    img_np = image.permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    ax.imshow(img_np)
    ax.set_title("Original Image", fontsize=14, fontweight="bold")
    ax.axis("off")
    
    # Ground truth (if provided)
    if ground_truth is not None:
        ax = fig.add_subplot(gs[0, 2:4])
        ax.imshow(img_np)
        ax.imshow(ground_truth.detach().cpu().numpy(), cmap="hot", alpha=0.5)
        ax.set_title("Ground Truth", fontsize=14, fontweight="bold")
        ax.axis("off")
    
    # Explanations
    for idx, (attr, label) in enumerate(zip(all_explanations, labels)):
        row = (idx + n_cols) // n_cols
        col = (idx + n_cols) % n_cols
        
        ax = fig.add_subplot(gs[row, col])
        
        # Aggregate attribution
        attr_agg = attr.abs().sum(dim=0).detach().cpu().numpy()
        attr_agg = (attr_agg - attr_agg.min()) / (attr_agg.max() - attr_agg.min() + 1e-8)
        
        # Overlay on image
        ax.imshow(img_np)
        im = ax.imshow(attr_agg, cmap="hot", alpha=0.6)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_agreement_heatmap(comparison_matrix, method_names, save_path, metric_name="Rank Agreement"):
    """
    Plot heatmap of agreement between methods.
    
    Args:
        comparison_matrix: (n_methods, n_methods) matrix
        method_names: List of method names
        save_path: Path to save figure
        metric_name: Name of metric
    """
    
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        comparison_matrix,
        xticklabels=method_names,
        yticklabels=method_names,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        square=True,
        cbar_kws={"label": metric_name}
    )
    
    plt.title(f"{metric_name} between XAI Methods", fontsize=16, fontweight="bold")
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Method", fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_metrics_comparison(summary_dict, save_path):
    """
    Plot bar chart comparing metrics across models/methods.
    
    Args:
        summary_dict: Dict[model_name][method_name] = metrics_dict
        save_path: Path to save figure
    """
    
    # Extract data
    models = []
    methods = []
    faithfulness_ins = []
    faithfulness_del = []
    sparsity = []
    
    for model_name, model_methods in summary_dict.items():
        for method_name, metrics_list in model_methods.items():
            if not metrics_list:
                continue
            
            models.append(model_name)
            methods.append(method_name)
            faithfulness_ins.append(np.mean([m["faithfulness_insertion"] for m in metrics_list]))
            faithfulness_del.append(np.mean([m["faithfulness_deletion"] for m in metrics_list]))
            sparsity.append(np.mean([m["sparsity"] for m in metrics_list]))
    
    # Create labels
    labels = [f"{m}\n{method}" for m, method in zip(models, methods)]
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(labels))
    width = 0.6
    
    # Faithfulness Insertion (higher = better)
    axes[0].bar(x, faithfulness_ins, width, color="steelblue")
    axes[0].set_ylabel("Faithfulness (Insertion)", fontsize=12)
    axes[0].set_title("Higher = Better", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[0].axhline(y=np.mean(faithfulness_ins), color="red", linestyle="--", label="Mean")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    
    # Faithfulness Deletion (lower = better)
    axes[1].bar(x, faithfulness_del, width, color="coral")
    axes[1].set_ylabel("Faithfulness (Deletion)", fontsize=12)
    axes[1].set_title("Lower = Better", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].axhline(y=np.mean(faithfulness_del), color="red", linestyle="--", label="Mean")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)
    
    # Sparsity (higher = simpler explanation)
    axes[2].bar(x, sparsity, width, color="mediumseagreen")
    axes[2].set_ylabel("Sparsity", fontsize=12)
    axes[2].set_title("Higher = Simpler", fontsize=14, fontweight="bold")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[2].axhline(y=np.mean(sparsity), color="red", linestyle="--", label="Mean")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)
    
    plt.suptitle("XAI Metrics Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confounder_analysis(confounded_metrics, clean_metrics, save_path):
    """
    Plot comparison of confounded vs. clean samples.
    
    Shows whether models rely on confounders.
    """
    
    models = list(confounded_metrics.keys())
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Faithfulness comparison
    conf_fid = [np.mean([m["faithfulness_insertion"] for m in confounded_metrics[model]["integrated_gradients"]]) 
                for model in models if "integrated_gradients" in confounded_metrics[model]]
    clean_fid = [np.mean([m["faithfulness_insertion"] for m in clean_metrics[model]["integrated_gradients"]]) 
                 for model in models if "integrated_gradients" in clean_metrics[model]]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0].bar(x - width/2, conf_fid, width, label="Confounded", color="orangered")
    axes[0].bar(x + width/2, clean_fid, width, label="Clean", color="dodgerblue")
    axes[0].set_ylabel("Faithfulness (Insertion)", fontsize=12)
    axes[0].set_title("Explanation Quality: Confounded vs. Clean", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=45, ha="right")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    
    # Delta faithfulness
    delta = np.array(conf_fid) - np.array(clean_fid)
    
    colors = ["red" if d > 0 else "green" for d in delta]
    axes[1].bar(x, delta, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color="black", linestyle="-", linewidth=1)
    axes[1].set_ylabel("Δ Faithfulness (Conf - Clean)", fontsize=12)
    axes[1].set_title("Positive = Worse on Confounded", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=45, ha="right")
    axes[1].grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
