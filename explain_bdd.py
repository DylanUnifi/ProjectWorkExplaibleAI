# explain_bdd.py

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import wandb
from pathlib import Path

from data_loader.bdd_oia_loader import get_bdd_oia_loaders
from models.temporal_qcnn import TemporalQCNN
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.grad_explainer import GradientExplainer
from explainability.metrics import XAIMetrics
from explainability.visualization import visualize_video_explanation
from utils.device import get_device

def explain_bdd_oia(config):
    """Generate explanations for BDD-OIA predictions."""
    
    DEVICE = get_device()
    
    # Load model
    print("🔧 Loading trained model...")
    checkpoint_path = config["model"]["checkpoint"]
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    
    model = TemporalQCNN(
        in_channels=3,
        n_qubits=config["quantum"]["n_qubits"],
        n_layers=config["quantum"]["layers"],
        backend=config["quantum"]["backend"],
        conv_channels=config["model"]["conv_channels"],
        hidden_sizes=config["model"]["hidden_sizes"],
        n_actions=4,
        n_explanations=21,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()
    
    # Load data
    print("📹 Loading BDD-OIA test data...")
    _, _, test_loader = get_bdd_oia_loaders(
        root_dir=config["dataset"]["root"],
        batch_size=config["batch_size"],
        num_workers=config.get("num_workers", 4),
        n_frames=config["dataset"]["n_frames"],
    )
    
    # Initialize explainers
    print("🔍 Initializing explainers...")
    
    # SHAP
    if config["methods"]["shap"]["enabled"]:
        print("  - SHAP")
        shap_explainer = SHAPExplainer(
            model,
            background_data=test_loader,
            method=config["methods"]["shap"]["method"]
        )
    
    # LIME (for individual frames)
    if config["methods"]["lime"]["enabled"]:
        print("  - LIME")
        lime_explainer = LIMEExplainer(model, n_classes=4)
    
    # Gradient methods
    if config["methods"]["gradient"]["enabled"]:
        print("  - Gradient-based (IG, Saliency, GradCAM)")
        grad_explainer = GradientExplainer(model)
    
    # Metrics
    metrics_evaluator = XAIMetrics(model, device=DEVICE)
    
    # W&B
    wandb.init(
        project="xai_qcnn",
        name=f"explain_bdd_{config['experiment_name']}",
        config=config
    )
    
    # Explain samples
    save_dir = Path(config["output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    n_samples = config.get("n_samples", 50)
    
    for i, batch in enumerate(tqdm(test_loader, total=n_samples, desc="Explaining")):
        if i >= n_samples:
            break
        
        videos = batch["video"].to(DEVICE)  # (B, C, T, H, W)
        actions = batch["action"]
        video_ids = batch["video_id"]
        
        # Take first video from batch
        video = videos[0:1]
        action = actions[0]
        video_id = video_ids[0]
        
        explanations = {}
        
        # ═══════════════════════════════════════════════════
        # SHAP
        # ═══════════════════════════════════════════════════
        if config["methods"]["shap"]["enabled"]:
            shap_attr = shap_explainer.explain(video, target_class=action)
            explanations["shap"] = shap_attr
            
            # Evaluate metrics
            metrics_shap = metrics_evaluator.evaluate_all(
                video, shap_attr,
                explainer=shap_explainer,
                target=action
            )
            
            wandb.log({
                f"shap/{video_id}/faithfulness_insertion": metrics_shap["faithfulness_insertion"],
                f"shap/{video_id}/faithfulness_deletion": metrics_shap["faithfulness_deletion"],
                f"shap/{video_id}/infidelity": metrics_shap["infidelity"],
                f"shap/{video_id}/sparsity": metrics_shap["sparsity"],
            })
        
        # ═══════════════════════════════════════════════════
        # Gradient methods
        # ═══════════════════════════════════════════════════
        if config["methods"]["gradient"]["enabled"]:
            grad_attrs = grad_explainer.explain_all(video, target=action)
            
            for method_name, attr in grad_attrs.items():
                if attr is not None:
                    explanations[method_name] = attr
                    
                    # Evaluate
                    metrics_grad = metrics_evaluator.evaluate_all(
                        video, attr, target=action
                    )
                    
                    wandb.log({
                        f"{method_name}/{video_id}/faithfulness_insertion": metrics_grad["faithfulness_insertion"],
                        f"{method_name}/{video_id}/faithfulness_deletion": metrics_grad["faithfulness_deletion"],
                    })
        
        # ═══════════════════════════════════════════════════
        # Visualize & Save
        # ═══════════════════════════════════════════════════
        for method_name, attr in explanations.items():
            # Visualize video with explanation overlay
            video_path = save_dir / f"{video_id}_{method_name}.mp4"
            
            visualize_video_explanation(
                video.cpu(),
                attr.cpu(),
                save_path=str(video_path),
                method_name=method_name,
                prediction=action.item()
            )
            
            # Log to W&B
            wandb.log({
                f"videos/{method_name}": wandb.Video(str(video_path))
            })
        
        results.append({
            "video_id": video_id,
            "action": action.item(),
            "methods": list(explanations.keys()),
        })
    
    # Summary
    print(f"\n✅ Generated explanations for {len(results)} videos")
    print(f"   Saved to: {save_dir}")
    
    wandb.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    explain_bdd_oia(config)
