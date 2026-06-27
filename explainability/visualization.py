# explainability/visualization.py

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio

def visualize_video_explanation(
    video,
    attribution,
    save_path,
    method_name="explanation",
    prediction=None,
    fps=5,
):
    """
    Overlay attribution heatmap on video frames.
    
    Args:
        video: (1, C, T, H, W) tensor
        attribution: (1, C, T, H, W) tensor
        save_path: Path to save video
        method_name: Name of explanation method
        prediction: Predicted class
        fps: Frames per second
    """
    video = video[0].permute(1, 2, 3, 0).detach().cpu().numpy()  # (T, H, W, C)
    attribution = attribution[0].abs().sum(dim=0).detach().cpu().numpy()  # (T, H, W)
    
    # Normalize attribution
    attr_normalized = (attribution - attribution.min()) / (attribution.max() - attribution.min() + 1e-8)
    
    frames = []
    
    for t in range(video.shape[0]):
        frame = video[t]  # (H, W, C)
        attr = attr_normalized[t]  # (H, W)
        
        # Convert to [0, 255]
        frame_uint8 = (frame * 255).astype(np.uint8)
        
        # Apply colormap to attribution
        heatmap = plt.colormaps["jet"](attr)[:, :, :3]  # (H, W, 3)
        heatmap = (heatmap * 255).astype(np.uint8)
        
        # Overlay
        overlay = cv2.addWeighted(frame_uint8, 0.6, heatmap, 0.4, 0)
        
        # Add text
        text = f"{method_name}"
        if prediction is not None:
            text += f" | Pred: {prediction}"
        
        cv2.putText(
            overlay, text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        frames.append(overlay)
    
    # Save video
    imageio.mimsave(save_path, frames, fps=fps)


def plot_attribution_comparison(
    image,
    attributions_dict,
    save_path,
    prediction=None,
    true_label=None,
):
    """
    Plot multiple attribution methods side-by-side.
    
    Args:
        image: (C, H, W) tensor
        attributions_dict: {"method_name": attribution_tensor}
        save_path: Path to save plot
    """
    n_methods = len(attributions_dict)
    
    fig, axes = plt.subplots(2, n_methods + 1, figsize=(4*(n_methods+1), 8), squeeze=False)
    
    # Original image
    img_np = image.permute(1, 2, 0).detach().cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
    
    axes[0, 0].imshow(img_np)
    title = "Original"
    if prediction is not None:
        title += f"\nPred: {prediction}"
    if true_label is not None:
        title += f"\nTrue: {true_label}"
    axes[0, 0].set_title(title)
    axes[0, 0].axis("off")
    
    axes[1, 0].axis("off")
    
    # Plot each attribution method
    for idx, (method_name, attr) in enumerate(attributions_dict.items(), start=1):
        # Positive attributions
        attr_pos = attr.sum(dim=0).detach().cpu().numpy()  # Sum over channels
        attr_pos = np.maximum(attr_pos, 0)
        attr_pos = (attr_pos - attr_pos.min()) / (attr_pos.max() - attr_pos.min() + 1e-8)
        
        # Row 0: signed attribution with diverging colormap (shows both positive and negative)
        attr_raw = attr.sum(dim=0).detach().cpu().numpy()
        abs_max = np.abs(attr_raw).max() + 1e-8
        axes[0, idx].imshow(img_np)
        im = axes[0, idx].imshow(attr_raw, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max, alpha=0.6)
        axes[0, idx].set_title(f"{method_name}\n(signed)")
        axes[0, idx].axis("off")
        plt.colorbar(im, ax=axes[0, idx], fraction=0.046)

        # Row 1: absolute importance (where prototypes fire, regardless of sign)
        attr_abs = np.abs(attr_raw)
        attr_abs = (attr_abs - attr_abs.min()) / (attr_abs.max() - attr_abs.min() + 1e-8)
        axes[1, idx].imshow(img_np)
        im = axes[1, idx].imshow(attr_abs, cmap="hot", alpha=0.6)
        axes[1, idx].set_title(f"{method_name}\n(magnitude)")
        axes[1, idx].axis("off")
        plt.colorbar(im, ax=axes[1, idx], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
