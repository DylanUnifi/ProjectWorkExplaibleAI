import os
import argparse
import torch
from tqdm import tqdm
import wandb

from dataset import get_clevr_hans_loaders, get_mnmath_loaders
from model import ResNet50Classifier, ViTClassifier, ProtoPNet, CLEVRQCNNClassifier
from explainability.shap_explainer import SHAPExplainer
from explainability.lime_explainer import LIMEExplainer
from explainability.metrics import XAIMetrics
from explainability.visualization import plot_attribution_comparison

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["resnet50", "vit", "protopnet", "hybrid_qcnn"])
    parser.add_argument("--dataset", type=str, required=True, choices=["clevr_hans3", "mnmath"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=32)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()

def load_model(args, n_classes, in_channels):
    if args.model == "resnet50":
        model = ResNet50Classifier(n_classes=n_classes, input_channels=in_channels)
    elif args.model == "vit":
        model = ViTClassifier(n_classes=n_classes, pretrained=False)
    elif args.model == "protopnet":
        model = ProtoPNet(num_classes=n_classes, num_prototypes=n_classes*10)
    elif args.model == "hybrid_qcnn":
        model = CLEVRQCNNClassifier(n_classes=n_classes, input_channel=in_channels, n_qubits=8, n_layers=1)

        
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
    if args.dataset == "clevr_hans3":
        train_loader, _, test_loader = get_clevr_hans_loaders(root_dir="./CLEVR-Hans3", batch_size=args.batch_size, max_samples=args.max_samples)
        n_classes = 3
        in_channels = 3
    else:
        train_loader, _, test_loader = get_mnmath_loaders(batch_size=args.batch_size, max_samples=args.max_samples)
        n_classes = 2
        in_channels = 1

    model = load_model(args, n_classes, in_channels)

    if args.model == "protopnet":
        print("ProtoPNet is self-explaining! No post-hoc explainer needed.")
        return

    print("Running SHAP and LIME Explainers...")
    shap_explainer = SHAPExplainer(model, train_loader, method="gradient")
    lime_explainer = LIMEExplainer(model, n_classes=n_classes)
    metrics_calc = XAIMetrics(model, args.device)

    for i, batch in enumerate(tqdm(test_loader, desc="Generating Explanations")):
        images, labels = batch["image"].to(args.device), batch["label"].to(args.device)
        
        # Get explanations
        shap_attrs = shap_explainer.explain(images, labels)
        lime_attrs = lime_explainer.explain(images, labels)
        
        # Calculate infidelity
        shap_infidelity = metrics_calc.infidelity(images, shap_attrs, labels, n_samples=10)
        lime_infidelity = metrics_calc.infidelity(images, lime_attrs, labels, n_samples=10)
        
        wandb.log({"shap_infidelity": shap_infidelity, "lime_infidelity": lime_infidelity})
        
        # Visualize first image in batch
        fig = plot_attribution_comparison(
            images[0].cpu().numpy(),
            shap_attrs[0],
            lime_attrs[0],
            title=f"Label: {labels[0].item()}"
        )
        wandb.log({f"Explanation_Batch_{i}": wandb.Image(fig)})
        
        import matplotlib.pyplot as plt
        plt.close(fig)

    wandb.finish()
    print("Explanations complete and logged to W&B!")

if __name__ == "__main__":
    main()
