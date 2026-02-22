# explain_comparative.py

"""
Comparative XAI analysis across all 5 models.
Generates explanations and evaluates quality metrics.
"""

import os
import yaml
import torch
import numpy as np
from pathlib import Path
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from data_loader.clevr_hans_loader import get_clevr_hans_loaders
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.grad_explainer import GradientExplainer
from explainability.metrics import XAIMetrics, ConfounderDetectionMetrics
from explainability.visualization import plot_attribution_comparison

def load_trained_models(config):
    """Load all trained models from checkpoints."""
    
    models = {}
    
    # Load each model
    for model_name in config["models_to_explain"]:
        checkpoint_path = Path(config["checkpoints"][model_name])
        
        if model_name == "quantum_kernel_svm":
            with open(checkpoint_path, "rb") as f:
                models[model_name] = pickle.load(f)
        
        else:
            # Neural network models
            checkpoint = torch.load(checkpoint_path)
            
            if model_name == "hybrid_qcnn":
                from models.clevr_qcnn import CLEVRQCNNClassifier
                model = CLEVRQCNNClassifier(
                    n_classes=config["n_classes"],
                    n_qubits=config["qcnn"]["n_qubits"],
                    n_layers=config["qcnn"]["layers"],
                )
            
            elif model_name == "resnet18":
                from models.resnet18_classifier import ResNet18Classifier
                model = ResNet18Classifier(n_classes=config["n_classes"])
            
            elif model_name == "vit":
                from models.vit_classifier import ViTClassifier
                model = ViTClassifier(n_classes=config["n_classes"])
            
            elif model_name == "protopnet":
                from models.protopnet import ProtoPNet
                model = ProtoPNet(n_classes=config["n_classes"])
            
            model.load_state_dict(checkpoint["model_state_dict"])
            model = model.to("cuda")
            model.eval()
            
            models[model_name] = model
    
    return models


def explain_single_sample(sample, models, explainers, config):
    """
    Generate all explanations for a single sample across all models.
    
    Returns:
        explanations: Dict[model_name][method_name] = attribution
    """
    
    image = sample["image"].unsqueeze(0).to("cuda")
    label = sample["label"].item()
    image_id = sample["image_id"]
    confounder_info = sample["confounder_info"]
    
    explanations = {}
    
    # ═══════════════════════════════════════════════════════════
    # 1. QUANTUM KERNEL SVM
    # ═══════════════════════════════════════════════════════════
    if "quantum_kernel_svm" in models:
        model_data = models["quantum_kernel_svm"]
        
        # SHAP (Kernel SHAP on features)
        if "shap" in explainers["quantum_kernel_svm"]:
            shap_values = explainers["quantum_kernel_svm"]["shap"].explain(
                image.cpu().numpy().flatten().reshape(1, -1)
            )
            explanations["quantum_kernel_svm"] = {
                "kernel_shap": torch.from_numpy(shap_values).view_as(image),
            }
    
    # ═══════════════════════════════════════════════════════════
    # 2-5. NEURAL NETWORK MODELS
    # ═══════════════════════════════════════════════════════════
    for model_name in ["hybrid_qcnn", "resnet18", "vit", "protopnet"]:
        if model_name not in models:
            continue
        
        model = models[model_name]
        explanations[model_name] = {}
        
        # Gradient-based methods
        if "gradient" in explainers[model_name]:
            grad_explainer = explainers[model_name]["gradient"]
            
            # Integrated Gradients
            ig_attr = grad_explainer.integrated_gradients(image, target=label)
            explanations[model_name]["integrated_gradients"] = ig_attr
            
            # GradCAM (if available)
            try:
                gradcam_attr = grad_explainer.gradcam(image, target=label)
                explanations[model_name]["gradcam"] = gradcam_attr
            except:
                pass
            
            # Saliency
            saliency_attr = grad_explainer.saliency_map(image, target=label)
            explanations[model_name]["saliency"] = saliency_attr
        
        # ViT-specific: Attention maps
        if model_name == "vit" and "attention" in explainers[model_name]:
            _, attentions = model(image, output_attentions=True)
            attention_map = model.attention_rollout(attentions)
            explanations[model_name]["attention_rollout"] = attention_map.unsqueeze(1)
        
        # ProtoPNet-specific: Prototype similarities
        if model_name == "protopnet":
            logits, distances = model(image, return_distances=True)
            explanations[model_name]["prototype_distances"] = distances
    
    return explanations, label, confounder_info


def evaluate_explanations(explanations, image, label, confounder_info, metrics_evaluator):
    """
    Evaluate quality of all explanations.
    
    Returns:
        results: Dict[model_name][method_name] = metrics_dict
    """
    
    results = {}
    
    for model_name, methods in explanations.items():
        results[model_name] = {}
        
        for method_name, attribution in methods.items():
            if method_name == "prototype_distances":
                continue  # Skip non-attribution outputs
            
            # Compute XAI quality metrics
            metrics = metrics_evaluator.evaluate_all(
                image,
                attribution,
                target=label
            )
            
            results[model_name][method_name] = metrics
    
    return results


def comparative_xai_study(config_path):
    """
    Main comparative XAI study.
    
    For each test sample:
    1. Generate explanations from all models
    2. Evaluate explanation quality
    3. Analyze confounder detection
    4. Visualize comparisons
    """
    
    # Load config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # W&B
    wandb.init(
        project="xai_comparative_study",
        name=f"explain_{config['experiment_name']}",
        config=config,
    )
    
    # Load data
    print("Loading CLEVR-Hans test data...")
    _, _, test_loader = get_clevr_hans_loaders(
        root_dir=config["dataset"]["root"],
        variant=config["dataset"]["variant"],
        batch_size=1,
        num_workers=8,
    )
    
    # Load trained models
    print("Loading trained models...")
    models = load_trained_models(config)
    
    # Initialize explainers
    print("Initializing explainers...")
    explainers = {}
    
    for model_name, model in models.items():
        explainers[model_name] = {}
        
        if model_name == "quantum_kernel_svm":
            # Kernel SHAP
            from explainability.kernel_shap_qsvm import QuantumKernelSHAP
            explainers[model_name]["shap"] = QuantumKernelSHAP(
                model["model"],
                model["K_train"],
                model["X_train"]
            )
        
        else:
            # Gradient explainer
            explainers[model_name]["gradient"] = GradientExplainer(model)
            
            if model_name == "vit":
                explainers[model_name]["attention"] = True
    
    # Metrics evaluator
    metrics_evaluator = XAIMetrics(list(models.values())[0], device="cuda")
    confounder_metrics = ConfounderDetectionMetrics()
    
    # Output directory
    save_dir = Path(config["output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Process samples
    all_results = {
        "confounded": [],
        "clean": [],
    }
    
    n_samples = config.get("n_samples", 100)
    
    for i, batch in enumerate(tqdm(test_loader, total=n_samples, desc="Explaining samples")):
        if i >= n_samples:
            break
        
        # Extract sample
        sample = {
            "image": batch["image"][0],
            "label": batch["label"][0],
            "confounder_info": batch["confounder_info"],
            "image_id": batch["image_id"][0],
        }
        
        is_confounded = sample["confounder_info"]["is_confounded"][0].item()
        
        # ═══════════════════════════════════════════════════════
        # Generate explanations
        # ═══════════════════════════════════════════════════════
        explanations, label, confounder_info = explain_single_sample(
            sample, models, explainers, config
        )
        
        # ═══════════════════════════════════════════════════════
        # Evaluate explanations
        # ═══════════════════════════════════════════════════════
        results = evaluate_explanations(
            explanations,
            sample["image"].unsqueeze(0).to("cuda"),
            label,
            confounder_info,
            metrics_evaluator
        )
        
        # Store results
        result_entry = {
            "image_id": sample["image_id"],
            "label": label,
            "is_confounded": is_confounded,
            "metrics": results,
        }
        
        if is_confounded:
            all_results["confounded"].append(result_entry)
        else:
            all_results["clean"].append(result_entry)
        
        # ═══════════════════════════════════════════════════════
        # Visualize (every 10th sample)
        # ═══════════════════════════════════════════════════════
        if i % 10 == 0:
            # Flatten explanations for visualization
            viz_explanations = {}
            for model_name, methods in explanations.items():
                for method_name, attr in methods.items():
                    if method_name != "prototype_distances":
                        viz_explanations[f"{model_name}_{method_name}"] = attr[0]
            
            # Plot comparison
            save_path = save_dir / f"comparison_{sample['image_id']}.png"
            plot_attribution_comparison(
                sample["image"],
                viz_explanations,
                save_path=str(save_path),
                prediction=label,
            )
            
            # Log to W&B
            wandb.log({
                f"visualizations/{sample['image_id']}": wandb.Image(str(save_path))
            })
    
    # ═══════════════════════════════════════════════════════════
    # Aggregate results & comparison
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("COMPARATIVE XAI RESULTS")
    print("="*60)
    
    # Compute average metrics per model/method
    summary = {}
    
    for split_name, results_list in all_results.items():
        print(f"\n{split_name.upper()} samples (n={len(results_list)}):")
        summary[split_name] = {}
        
        # Aggregate per model/method
        for model_name in models.keys():
            summary[split_name][model_name] = {}
            
            for entry in results_list:
                if model_name not in entry["metrics"]:
                    continue
                
                for method_name, metrics in entry["metrics"][model_name].items():
                    if method_name not in summary[split_name][model_name]:
                        summary[split_name][model_name][method_name] = []
                    
                    summary[split_name][model_name][method_name].append(metrics)
        
        # Compute averages
        for model_name, methods in summary[split_name].items():
            print(f"\n  {model_name}:")
            
            for method_name, metrics_list in methods.items():
                if not metrics_list:
                    continue
                
                avg_faithfulness_ins = np.mean([m["faithfulness_insertion"] for m in metrics_list])
                avg_faithfulness_del = np.mean([m["faithfulness_deletion"] for m in metrics_list])
                avg_sparsity = np.mean([m["sparsity"] for m in metrics_list])
                
                print(f"    {method_name:20s}: Fid_ins={avg_faithfulness_ins:.3f}, Fid_del={avg_faithfulness_del:.3f}, Sparsity={avg_sparsity:.3f}")
                
                # Log to W&B
                wandb.log({
                    f"{split_name}/{model_name}_{method_name}/faithfulness_insertion": avg_faithfulness_ins,
                    f"{split_name}/{model_name}_{method_name}/faithfulness_deletion": avg_faithfulness_del,
                    f"{split_name}/{model_name}_{method_name}/sparsity": avg_sparsity,
                })
    
    # ═══════════════════════════════════════════════════════════
    # Confounder detection comparison
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("CONFOUNDER DETECTION")
    print("="*60)
    
    # Compare: Are confounded samples explained differently?
    for model_name in models.keys():
        print(f"\n{model_name}:")
        
        # Get average metrics for confounded vs clean
        for method_name in summary["confounded"].get(model_name, {}).keys():
            conf_metrics = summary["confounded"][model_name][method_name]
            clean_metrics = summary["clean"].get(model_name, {}).get(method_name, [])
            
            if not conf_metrics or not clean_metrics:
                continue
            
            conf_fid_ins = np.mean([m["faithfulness_insertion"] for m in conf_metrics])
            clean_fid_ins = np.mean([m["faithfulness_insertion"] for m in clean_metrics])
            
            diff = conf_fid_ins - clean_fid_ins
            
            print(f"  {method_name:20s}: Δ Faithfulness = {diff:+.3f}")
            
            wandb.log({
                f"confounder_analysis/{model_name}_{method_name}_delta_faithfulness": diff,
            })
    
    # Save summary
    with open(save_dir / "summary.pkl", "wb") as f:
        pickle.dump(summary, f)
    
    print(f"\n✅ Results saved to: {save_dir}")
    
    wandb.finish()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    comparative_xai_study(args.config)
