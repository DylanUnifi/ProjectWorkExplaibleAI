import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from dataset import get_cle4evr_loaders, get_mnmath_loaders
from model import ResNet50Classifier, ViTClassifier, ProtoPNet, HybridQCNNClassifier, HybridQViT
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.gradcam_explainer import GradCAMExplainer
from explainability.rollout_explainer import RolloutExplainer
from explainability.metrics import XAIMetrics
from explainability.advanced_metrics import ComparativeXAIMetrics

from explainability.advanced_visualization import plot_comparison_grid, plot_agreement_heatmap, plot_metrics_comparison

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "vit", "protopnet", "hybrid_qcnn", "hybrid_qvit"])
    parser.add_argument("--dataset", type=str, required=True, choices=["cle4evr", "mnmath"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=128, help="Number of samples to explain")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
        if logits.dim() > 2:
            return logits[:, 0, :]  # Explain the first equation by default
        return logits

def load_model(args, n_classes, in_channels, num_equations=1, num_concepts=0):
    kwargs = {"num_equations": num_equations, "num_concepts": num_concepts}
    if args.model == "resnet50":
        model = ResNet50Classifier(n_classes=n_classes, input_channels=in_channels, **kwargs)
    elif args.model == "vit":
        model = ViTClassifier(n_classes=n_classes, pretrained=True, input_channels=in_channels, **kwargs)
    elif args.model == "protopnet":
        model = ProtoPNet(n_classes=n_classes, input_channels=in_channels, n_prototypes_per_class=10, **kwargs)
    elif args.model == "hybrid_qcnn":
        # Retour à 8 qubits pour éviter le surapprentissage (overfitting massif à 12 qubits)
        model = HybridQCNNClassifier(n_classes=n_classes, input_channel=in_channels, n_qubits=8, n_layers=1, backend="lightning.qubit", **kwargs)
    elif args.model == "hybrid_qvit":
        model = HybridQViT(n_classes=n_classes, input_channel=in_channels, n_qubits=8, img_size=64, patch_size=8, backend="lightning.qubit", **kwargs)

        
    ckpt_path = f"checkpoints/{args.model}_{args.dataset}_best.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        print(f"Loaded weights from {ckpt_path}")
    else:
        print(f"Warning: No weights found at {ckpt_path}. Using random weights.")
        
    model = model.to(args.device)
    model.eval()
    return model

def main():
    args = parse_args()
    wandb.init(project="XAI_Comparative_Study", name=f"Explain_{args.model}_{args.dataset}")

    print(f"Loading {args.dataset}...")
    num_equations = 1
    num_concepts = 0
    if args.dataset == "cle4evr":
        train_loader, _, test_loader = get_cle4evr_loaders(root_dir="./CLEVR-Hans3", batch_size=args.batch_size, max_samples=args.max_samples)
        n_classes = 2
        in_channels = 3
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        train_loader, _, test_loader, num_equations, num_concepts = get_mnmath_loaders(batch_size=args.batch_size, max_samples=args.max_samples)
        n_classes = 19
        in_channels = 1
        mean = [0.5]
        std = [0.5]

    model = load_model(args, n_classes, in_channels, num_equations, num_concepts)

    if args.model == "protopnet":
        print("ProtoPNet is self-explaining! No post-hoc explainer needed.")
        return

    print("Initializing Explainers...")
    wrapped_model = ModelWrapper(model)
    metrics_calc = XAIMetrics(wrapped_model, args.device)
    
    explainers = {
        "SHAP": SHAPExplainer(wrapped_model, train_loader, method="gradient"),
        "LIME": LIMEExplainer(wrapped_model, n_classes=n_classes, mean=mean, std=std)
    }
    
    # Dynamically add architecture-specific explainers
    if hasattr(model, 'generate_gradcam'):
        print("Model supports GradCAM. Adding GradCAMExplainer...")
        explainers["GradCAM"] = GradCAMExplainer(wrapped_model)
    if hasattr(model, 'attention_rollout'):
        print("Model supports Attention Rollout. Adding RolloutExplainer...")
        explainers["Rollout"] = RolloutExplainer(wrapped_model)
        
    # Initialize accumulators dynamically based on active explainers
    avg_metrics = {}
    for name in explainers.keys():
        avg_metrics[f"{name.lower()}_infidelity"] = 0.0
        avg_metrics[f"{name.lower()}_complexity"] = 0.0
        
    avg_metrics["rank_corr_shap_vs_lime"] = 0.0
    avg_metrics["top_k_overlap_shap_vs_lime"] = 0.0
    
    num_batches = 0
    
    summary_dict = {
        args.model: {name: [] for name in explainers.keys()}
    }
    all_comparison_matrices = []

    for i, batch in enumerate(tqdm(test_loader, desc="Generating Explanations")):
        images, labels = batch["image"].to(args.device), batch["label"].to(args.device)
        
        # For evaluation, take the first label if there are multiple equations
        eval_labels = labels[:, 0] if labels.dim() > 1 else labels
        
        # Get explanations dynamically
        explanations_dict = {}
        batch_log_metrics = {}
        
        for name, explainer in explainers.items():
            attrs = explainer.explain(images, eval_labels)
            explanations_dict[name] = attrs
            
            # Calculate metrics
            infid = metrics_calc.infidelity(images, attrs, eval_labels, n_samples=10)
            comp = ComparativeXAIMetrics.explanation_complexity(attrs)
            
            summary_dict[args.model][name].append({"infidelity": infid, "complexity": comp})
            
            batch_log_metrics[f"{name.lower()}_infidelity"] = infid
            batch_log_metrics[f"{name.lower()}_complexity"] = comp
            
            avg_metrics[f"{name.lower()}_infidelity"] += infid
            avg_metrics[f"{name.lower()}_complexity"] += comp
            
        # Hardcoded SHAP vs LIME for continuity
        rank_corr = ComparativeXAIMetrics.rank_agreement(explanations_dict["SHAP"], explanations_dict["LIME"], method="spearman")
        top_k_over = ComparativeXAIMetrics.top_k_overlap(explanations_dict["SHAP"], explanations_dict["LIME"], k=100)
        
        batch_log_metrics["rank_corr_shap_vs_lime"] = rank_corr
        batch_log_metrics["top_k_overlap_shap_vs_lime"] = top_k_over
        avg_metrics["rank_corr_shap_vs_lime"] += rank_corr
        avg_metrics["top_k_overlap_shap_vs_lime"] += top_k_over
        
        # Matrix over ALL methods
        comparison_matrix, methods = ComparativeXAIMetrics.compare_all_methods(explanations_dict, metrics_to_compute=["rank_agreement"])
        all_comparison_matrices.append(comparison_matrix["rank_agreement"])
        
        wandb.log(batch_log_metrics)
        
        import os
        os.makedirs("explanations", exist_ok=True)
        save_path = f"explanations/{args.model}_{args.dataset}_batch_{i}_grid.png"
        
        # Visualize first image in batch using comparison grid
        first_img_attrs = {name: attrs[0] for name, attrs in explanations_dict.items()}
        plot_comparison_grid(
            images[0],
            {args.model: first_img_attrs},
            save_path=save_path
        )
        wandb.log({f"Explanation_Batch_{i}": wandb.Image(save_path)})
        
        num_batches += 1

    # Print final averages
    if num_batches > 0:
        print("\n" + "="*50)
        print(f"AVERAGE XAI METRICS OVER {num_batches} BATCHES ({args.max_samples} samples max)")
        print("="*50)
        for k in avg_metrics.keys():
            avg_metrics[k] /= num_batches
            print(f"{k}: {avg_metrics[k]:.5f}")
        print("="*50)
        
        # Log averages to W&B
        wandb.log({f"avg_{k}": v for k, v in avg_metrics.items()})
        
        import os
        import numpy as np
        os.makedirs("explanations", exist_ok=True)
        
        # 1. Plot Agreement Heatmap
        if all_comparison_matrices:
            avg_matrix = np.mean(all_comparison_matrices, axis=0)
            heatmap_path = f"explanations/{args.model}_{args.dataset}_agreement_heatmap.png"
            plot_agreement_heatmap(avg_matrix, methods, save_path=heatmap_path)
            wandb.log({"Agreement_Heatmap": wandb.Image(heatmap_path)})
        
        # 2. Plot Metrics Comparison
        metrics_plot_path = f"explanations/{args.model}_{args.dataset}_metrics_comparison.png"
        plot_metrics_comparison(summary_dict, save_path=metrics_plot_path)
        wandb.log({"Metrics_Comparison": wandb.Image(metrics_plot_path)})

    wandb.finish()
    print("Explanations complete and logged to W&B!")

if __name__ == "__main__":
    main()
