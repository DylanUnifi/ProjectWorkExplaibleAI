# explain_clevr.py

import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from data_loader.clevr_hans_loader import get_clevr_hans_loaders
from models.clevr_qcnn import CLEVRQCNNClassifier
from explainability.shap_explainer import SHAPExplainer
from explainability.grad_explainer import GradientExplainer
from explainability.metrics import XAIMetrics, ConfounderDetectionMetrics
from utils.device import get_device


def load_model_for_explanation(model_name, config, n_classes, device):
    """
    Load a single model by name using the multi-model config schema.

    Returns (model, is_torch_module) where is_torch_module is False for
    non-differentiable models (e.g. quantum_kernel_svm) that cannot use
    gradient-based XAI methods.
    """
    checkpoint_path = config["checkpoints"].get(model_name)
    if checkpoint_path is None:
        print(f"⚠️  No checkpoint configured for {model_name} — skipping.")
        return None, False
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint not found for {model_name}: {checkpoint_path} — skipping.")
        return None, False

    if model_name == "quantum_kernel_svm":
        with open(checkpoint_path, "rb") as f:
            model = pickle.load(f)
        return model, False

    # --- Neural network models ---
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    if model_name == "hybrid_qcnn":
        qcnn_cfg = config["qcnn"]
        model = CLEVRQCNNClassifier(
            n_classes=n_classes,
            input_channel=3,
            n_qubits=qcnn_cfg["n_qubits"],
            n_layers=qcnn_cfg["layers"],
            backend=qcnn_cfg["backend"],
            conv_channels=qcnn_cfg.get("conv_channels"),
            hidden_sizes=qcnn_cfg.get("hidden_sizes"),
            dropout=qcnn_cfg.get("dropout", 0.0),
        )

    elif model_name == "resnet18":
        from models.resnet18_classifier import ResNet18Classifier
        model = ResNet18Classifier(n_classes=n_classes, pretrained=False)

    elif model_name == "vit":
        from models.vit_classifier import ViTClassifier
        model = ViTClassifier(n_classes=n_classes, pretrained=False)

    elif model_name == "protopnet":
        try:
            from models.protopnet import ProtoPNet
            model = ProtoPNet(n_classes=n_classes)
        except Exception as exc:
            print(f"⚠️  Could not load ProtoPNet ({exc}) — skipping.")
            return None, False

    else:
        print(f"⚠️  Unknown model type '{model_name}' — skipping.")
        return None, False

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, True


def _explain_model(model_name, model, is_torch_module, test_loader, n_samples, config, save_dir, device):
    """Run XAI analysis for a single model and return confounded/non-confounded results."""

    confounded_results = []
    non_confounded_results = []

    if not is_torch_module:
        print(f"⏭️  Skipping gradient-based XAI for {model_name} (not a torch.nn.Module).")
        return confounded_results, non_confounded_results

    grad_explainer = GradientExplainer(model)
    shap_explainer = SHAPExplainer(model, background_data=test_loader, method="deep")
    xai_metrics = XAIMetrics(model, device=device)

    for i, batch in enumerate(tqdm(test_loader, total=n_samples, desc=f"Explaining {model_name}")):
        if i >= n_samples:
            break

        image = batch["image"].to(device)
        label = batch["label"]
        confounder_info = batch["confounder_info"]
        image_id = batch["image_id"][0]

        is_confounded = confounder_info[0]["is_confounded"]
        confounder_attrs = confounder_info[0].get("confounder_attrs", [])

        # ═══════════════════════════════════════════════════
        # Generate explanations
        # ═══════════════════════════════════════════════════
        explanations = {}

        ig_attr = grad_explainer.integrated_gradients(image, target=label)
        explanations["integrated_gradients"] = ig_attr

        gradcam_attr = grad_explainer.gradcam(image, target=label)
        explanations["gradcam"] = gradcam_attr

        if i % 5 == 0:
            shap_attr = shap_explainer.explain(image, target_class=label)
            explanations["shap"] = shap_attr

        # ═══════════════════════════════════════════════════
        # Evaluate metrics
        # ═══════════════════════════════════════════════════
        for method_name, attr in explanations.items():
            if not isinstance(attr, torch.Tensor):
                attr = torch.tensor(np.array(attr), dtype=torch.float32)
            attr = attr.detach().to(device=image.device, dtype=torch.float32)
            if attr.dim() == 5:
                tc = int(label.item()) if hasattr(label, "item") else int(label)
                attr = attr[..., tc]
            if attr.dim() == 3:
                attr = attr.unsqueeze(0)

            metrics = xai_metrics.evaluate_all(image, attr, target=label)

            result = {
                "image_id": image_id,
                "method": method_name,
                "is_confounded": int(is_confounded),
                "faithfulness_insertion": metrics["faithfulness_insertion"],
                "faithfulness_deletion": metrics["faithfulness_deletion"],
                "infidelity": metrics["infidelity"],
                "sparsity": metrics["sparsity"],
            }

            wandb.log({
                f"{method_name}/{'confounded' if is_confounded else 'clean'}/faithfulness_insertion": metrics["faithfulness_insertion"],
                f"{method_name}/{'confounded' if is_confounded else 'clean'}/faithfulness_deletion": metrics["faithfulness_deletion"],
            })

            if is_confounded:
                confounded_results.append(result)
            else:
                non_confounded_results.append(result)

        # ═══════════════════════════════════════════════════
        # Visualize
        # ═══════════════════════════════════════════════════
        fig, axes = plt.subplots(2, len(explanations) + 1, figsize=(4*(len(explanations)+1), 8))

        img_np = image[0].permute(1, 2, 0).detach().cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Original")
        axes[0, 0].axis("off")

        axes[1, 0].text(0.5, 0.5,
                        f"Confounded: {int(is_confounded)}\nLabel: {label.item()}",
                        ha="center", va="center")
        axes[1, 0].axis("off")

        for idx, (method_name, attr) in enumerate(explanations.items(), start=1):
            attr_pos = attr[0].sum(dim=0).detach().cpu().numpy()
            attr_pos = np.maximum(attr_pos, 0)

            axes[0, idx].imshow(img_np)
            im = axes[0, idx].imshow(attr_pos, cmap="hot", alpha=0.6)
            axes[0, idx].set_title(f"{method_name} (pos)")
            axes[0, idx].axis("off")
            plt.colorbar(im, ax=axes[0, idx])

            attr_neg = np.minimum(attr[0].sum(dim=0).detach().cpu().numpy(), 0)

            axes[1, idx].imshow(img_np)
            im = axes[1, idx].imshow(-attr_neg, cmap="cool", alpha=0.6)
            axes[1, idx].set_title(f"{method_name} (neg)")
            axes[1, idx].axis("off")
            plt.colorbar(im, ax=axes[1, idx])

        plt.tight_layout()
        save_path = save_dir / f"{model_name}_{image_id}_explanations.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        wandb.log({f"images/{model_name}/{image_id}": wandb.Image(str(save_path))})

    return confounded_results, non_confounded_results


def explain_clevr_hans(config):
    """
    Generate explanations for CLEVR-Hans and analyze
    whether model relies on confounders vs. true class rules.

    Supports both single-model configs (legacy, with ``model`` and ``quantum``
    top-level keys) and multi-model configs (with ``models_to_explain`` and
    ``checkpoints`` top-level keys).
    """

    DEVICE = get_device()
    variant = config["dataset"]["variant"]

    # ── Backward-compatibility: promote single-model config to multi-model ──
    if "models_to_explain" not in config:
        # Legacy single-model config
        n_classes = config.get("n_classes", 3 if variant == "clevr_hans3" else 7)
        config = dict(config)  # shallow copy so we don't mutate caller's dict
        config["n_classes"] = n_classes
        config.setdefault("checkpoints", {})
        config["checkpoints"]["hybrid_qcnn"] = config["model"]["checkpoint"]
        config["models_to_explain"] = ["hybrid_qcnn"]
        # Map legacy `quantum` section to `qcnn`
        if "quantum" in config and "qcnn" not in config:
            qcnn = dict(config["quantum"])
            qcnn.setdefault("conv_channels", config["model"].get("conv_channels"))
            qcnn.setdefault("hidden_sizes", config["model"].get("hidden_sizes"))
            qcnn.setdefault("dropout", config["model"].get("dropout", 0.0))
            config["qcnn"] = qcnn

    n_classes = config.get("n_classes", 3 if variant == "clevr_hans3" else 7)

    # Load data (shared across models)
    print(f"🧩 Loading {variant} test data...")
    _, _, test_loader = get_clevr_hans_loaders(
        root_dir=config["dataset"]["root"],
        variant=variant,
        batch_size=1,
        num_workers=config.get("num_workers", config["dataset"].get("num_workers", 4)),
    )

    # W&B
    wandb.init(
        project="xai_qcnn",
        name=f"explain_clevr_{config['experiment_name']}",
        config=config,
    )

    save_dir = Path(config["output_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    n_samples = config.get("n_samples", 100)

    all_confounded = []
    all_non_confounded = []

    for model_name in config["models_to_explain"]:
        print(f"\n🔧 Loading model: {model_name} ...")
        model, is_torch_module = load_model_for_explanation(model_name, config, n_classes, DEVICE)
        if model is None:
            continue

        conf, non_conf = _explain_model(
            model_name, model, is_torch_module,
            test_loader, n_samples, config, save_dir, DEVICE,
        )
        all_confounded.extend(conf)
        all_non_confounded.extend(non_conf)

    confounded_results = all_confounded
    non_confounded_results = all_non_confounded

    # ═══════════════════════════════════════════════════
    # Compare confounded vs. non-confounded
    # ═══════════════════════════════════════════════════
    print("\n📊 Comparing confounded vs. non-confounded classes:")
    
    for method in ["integrated_gradients", "gradcam"]:
        conf_results = [r for r in confounded_results if r["method"] == method]
        clean_results = [r for r in non_confounded_results if r["method"] == method]
        
        if conf_results and clean_results:
            conf_ins = [r["faithfulness_insertion"] for r in conf_results if "faithfulness_insertion" in r]
            clean_ins = [r["faithfulness_insertion"] for r in clean_results if "faithfulness_insertion" in r]
            conf_del = [r["faithfulness_deletion"] for r in conf_results if "faithfulness_deletion" in r]
            clean_del = [r["faithfulness_deletion"] for r in clean_results if "faithfulness_deletion" in r]

            if not (conf_ins and clean_ins and conf_del and clean_del):
                continue

            conf_fid_ins = np.mean(conf_ins)
            clean_fid_ins = np.mean(clean_ins)
            conf_fid_del = np.mean(conf_del)
            clean_fid_del = np.mean(clean_del)

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
