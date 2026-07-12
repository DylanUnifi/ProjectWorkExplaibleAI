import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

from dataset import get_cle4evr_loaders, get_clevr_hans_loaders, get_mnmath_loaders
from model import ResNet50Classifier, ViTClassifier, ProtoPNet, HybridQCNNClassifier
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.metrics import XAIMetrics
from explainability.visualization import plot_attribution_comparison

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "vit", "protopnet", "hybrid_qcnn"])
    parser.add_argument("--dataset", type=str, required=True, choices=["cle4evr", "mnmath"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=32)
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
        model = HybridQCNNClassifier(n_classes=n_classes, input_channel=in_channels, n_qubits=16, n_layers=2, backend="lightning.gpu", **kwargs)

        
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
    else:
        train_loader, _, test_loader, num_equations, num_concepts = get_mnmath_loaders(batch_size=args.batch_size, max_samples=args.max_samples)
        n_classes = 19
        in_channels = 1

    model = load_model(args, n_classes, in_channels, num_equations, num_concepts)

    if args.model == "protopnet":
        print("ProtoPNet is self-explaining! No post-hoc explainer needed.")
        return

    print("Running SHAP and LIME Explainers...")
    wrapped_model = ModelWrapper(model)
    shap_explainer = SHAPExplainer(wrapped_model, train_loader, method="gradient")
    lime_explainer = LIMEExplainer(wrapped_model, n_classes=n_classes)
    metrics_calc = XAIMetrics(wrapped_model, args.device)

    for i, batch in enumerate(tqdm(test_loader, desc="Generating Explanations")):
        images, labels = batch["image"].to(args.device), batch["label"].to(args.device)
        
        # For evaluation, take the first label if there are multiple equations
        eval_labels = labels[:, 0] if labels.dim() > 1 else labels
        
        # Get explanations
        shap_attrs = shap_explainer.explain(images, eval_labels)
        lime_attrs = lime_explainer.explain(images, eval_labels)
        
        # Calculate infidelity
        shap_infidelity = metrics_calc.infidelity(images, shap_attrs, eval_labels, n_samples=10)
        lime_infidelity = metrics_calc.infidelity(images, lime_attrs, eval_labels, n_samples=10)
        
        wandb.log({"shap_infidelity": shap_infidelity, "lime_infidelity": lime_infidelity})
        
        import os
        os.makedirs("explanations", exist_ok=True)
        save_path = f"explanations/{args.model}_{args.dataset}_batch_{i}.png"
        
        # Visualize first image in batch
        plot_attribution_comparison(
            images[0],
            {"SHAP": shap_attrs[0], "LIME": lime_attrs[0]},
            save_path=save_path,
            true_label=eval_labels[0].item()
        )
        wandb.log({f"Explanation_Batch_{i}": wandb.Image(save_path)})

    wandb.finish()
    print("Explanations complete and logged to W&B!")

if __name__ == "__main__":
    main()
