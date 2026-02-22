# explain_clevr.py

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from pathlib import Path

from data_loader.clevr_hans_loader import get_clevr_hans_loaders
from models.clevr_qcnn import CLEVRQCNNClassifier
from explainability.shap_explainer import SHAPExplainer
from explainability.grad_explainer import GradientExplainer
from explainability.metrics import XAIMetrics, ConfounderDetectionMetrics
from utils.device import get_device

def explain_clevr_hans(config):
    """
    Generate explanations for CLEVR-Hans and analyze
    whether model relies on confounders vs. true class rules.
    """
    
    DEVICE = get_device()
    variant = config["dataset"]["variant"]
    n_classes = 3 if variant == "clevr_hans3" else 7
    
    # Load model
    print("🔧 Loading trained model...")
    checkpoint_path = config["model"]["checkpoint"]
    checkpoint = torch.load(checkpoint_path)
    
    model = CLEVRQCNNClassifier(
    n_classes=n_classes,
    input_channel=3,
    n_qubits=config["quantum"]["n_qubits"],
    n_layers=config["quantum"]["layers"],
    backend=config["quantum"]["backend"],
    conv_channels=config["model"]["conv_channels"],
    hidden_sizes=config["model"]["hidden_sizes"],
    dropout=config["model"]["dropout"],
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    
    # Load data
    print(f"🧩 Loading {variant} test data...")
    _, _, test_loader = get_clevr_hans_loaders(
        root_dir=config["dataset"]["root"],
        variant=variant,
        batch_size=1,  # Process one at a time for LIME
        num_workers=config.get("num_workers", 4),
    )
    
    # Explainers
    print("🔍 Initializing explainers...")
    grad_explainer = GradientExplainer(model)
    shap_explainer = SHAPExplainer(model, background_data=test_loader, method="deep")
    
    # Metrics
    xai_metrics = XAIMetrics(model, device=DEVICE)
    confounder_metrics = ConfounderDetectionMetrics()
    
    # W&B
    wandb.init(
        project="xai_qcnn",
        name=f"explain_clevr_{config['experiment_name']}",
        config=config
    )
    
    save_dir = Path(config["output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze confounded vs. non-confounded classes
    confounded_results = []
    non_confounded_results = []
    
    n_samples = config.get("n_samples", 100)
    
    for i, batch in enumerate(tqdm(test_loader, total=n_samples, desc="Explaining")):
        if i >= n_samples:
            break
        
        image = batch["image"].to(DEVICE)
        label = batch["label"]
        confounder_info = batch["confounder_info"]
        image_id = batch["image_id"][0]
        
        is_confounded = confounder_info[0]["is_confounded"]
        confounder_attrs = confounder_info[0].get("confounder_attrs", [])
        
        # ═══════════════════════════════════════════════════
        # Generate explanations
        # ═══════════════════════════════════════════════════
        explanations = {}
        
        # Integrated Gradients
        ig_attr = grad_explainer.integrated_gradients(image, target=label)
        explanations["integrated_gradients"] = ig_attr
        
        # GradCAM
        gradcam_attr = grad_explainer.gradcam(image, target=label)
        explanations["gradcam"] = gradcam_attr
        
        # SHAP (expensive, optionally skip some)
        if i % 5 == 0:
            shap_attr = shap_explainer.explain(image, target_class=label)
            explanations["shap"] = shap_attr
        
        # ═══════════════════════════════════════════════════
        # Evaluate metrics
        # ═══════════════════════════════════════════════════
        for method_name, attr in explanations.items():
            print("image shape:", image.shape, image.dtype, image.device)
            print("attr shape:", attr.shape, attr.dtype, attr.device)
            print("label:", label, type(label))
            # Ensure attribution is BCHW float32 on same device as image
            if attr.dim() == 5:
                # likely BHWC or BCHW with extra dim -> reduce last dim
                attr = attr.mean(dim=-1)
            if attr.dim() == 3:
                attr = attr.unsqueeze(0)
            attr = attr.to(device=image.device, dtype=torch.float32)
            # normalize attr to BCHW float32 on cuda
            attr = torch.as_tensor(attr)
            if attr.dim() == 5:
                # most likely (B,C,H,W,classes) -> pick target class
                tc = int(label.item()) if hasattr(label, "item") else int(label)
                attr = attr[..., tc]
            if attr.dim() == 4 and attr.shape[-1] in (1, 3):  # BHWC -> BCHW
                # if last dim looks like channels, permute
                # (this is a heuristic; safer to do in shap_explainer)
                pass
            if attr.dim() == 3:
                attr = attr.unsqueeze(0)
            attr = attr.to(device=image.device, dtype=torch.float32)
            #metrics = xai_metrics.evaluate_all(image, attr, target=label)
            
            # Store results
            result = {
                "image_id": image_id,
                "method": method_name,
                "is_confounded": int(is_confounded),
                #"faithfulness_insertion": metrics["faithfulness_insertion"],
                #"faithfulness_deletion": metrics["faithfulness_deletion"],
                #"infidelity": metrics["infidelity"],
                #"sparsity": metrics["sparsity"],
            }
            
            if is_confounded:
                confounded_results.append(result)
            else:
                non_confounded_results.append(result)
            
            # Log to W&B
            #wandb.log({
            #    f"{method_name}/{'confounded' if is_confounded else 'clean'}/faithfulness_insertion": metrics["faithfulness_insertion"],
            #    f"{method_name}/{'confounded' if is_confounded else 'clean'}/faithfulness_deletion": metrics["faithfulness_deletion"],
            #})
        
        # ═══════════════════════════════════════════════════
        # Visualize
        # ═══════════════════════════════════════════════════
        fig, axes = plt.subplots(2, len(explanations) + 1, figsize=(4*(len(explanations)+1), 8))
        
        # Original image
        img_np = image[0].permute(1, 2, 0).detach().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())
        
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")
        
        axes[1, 0].text(0.5, 0.5, 
                       f"Confounded: {int(is_confounded)}\nLabel: {label.item()}",
                       ha="center", va="center")
        axes[1, 0].axis("off")
        
        # Explanations
        for idx, (method_name, attr) in enumerate(explanations.items(), start=1):
            # Positive attributions
            attr_pos = attr[0].sum(dim=0).detach().cpu().numpy()  # Sum over channels
            attr_pos = np.maximum(attr_pos, 0)
            
            axes[0, idx].imshow(img_np)
            im = axes[0, idx].imshow(attr_pos, cmap="hot", alpha=0.6)
            axes[0, idx].set_title(f"{method_name} (pos)")
            axes[0, idx].axis("off")
            plt.colorbar(im, ax=axes[0, idx])
            
            # Negative attributions
            attr_neg = np.minimum(attr[0].sum(dim=0).detach().cpu().numpy(), 0)
            
            axes[1, idx].imshow(img_np)
            im = axes[1, idx].imshow(-attr_neg, cmap="cool", alpha=0.6)
            axes[1, idx].set_title(f"{method_name} (neg)")
            axes[1, idx].axis("off")
            plt.colorbar(im, ax=axes[1, idx])
        
        plt.tight_layout()
        save_path = save_dir / f"{image_id}_explanations.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        
        # Log to W&B
        wandb.log({f"images/{image_id}": wandb.Image(str(save_path))})
    
    # ═══════════════════════════════════════════════════
    # Compare confounded vs. non-confounded
    # ═══════════════════════════════════════════════════
    print("\n📊 Comparing confounded vs. non-confounded classes:")
    
    for method in ["integrated_gradients", "gradcam"]:
        conf_results = [r for r in confounded_results if r["method"] == method]
        clean_results = [r for r in non_confounded_results if r["method"] == method]
        
        if conf_results and clean_results:
            conf_fid_ins = np.mean([r["faithfulness_insertion"] for r in conf_results])
            clean_fid_ins = np.mean([r["faithfulness_insertion"] for r in clean_results])
            
            conf_fid_del = np.mean([r["faithfulness_deletion"] for r in conf_results])
            clean_fid_del = np.mean([r["faithfulness_deletion"] for r in clean_results])
            
            print(f"\n{method}:")
            print(f"  Confounded   - Insertion: {conf_fid_ins:.4f}, Deletion: {conf_fid_del:.4f}")
            print(f"  Non-confounded - Insertion: {clean_fid_ins:.4f}, Deletion: {clean_fid_del:.4f}")
            
            wandb.log({
                f"summary/{method}/confounded_faithfulness_insertion": conf_fid_ins,
                f"summary/{method}/clean_faithfulness_insertion": clean_fid_ins,
                f"summary/{method}/confounded_faithfulness_deletion": conf_fid_del,
                f"summary/{method}/clean_faithfulness_deletion": clean_fid_del,
            })
    
    wandb.finish()
    print(f"\n✅ Explanations saved to: {save_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    explain_clevr_hans(config)
