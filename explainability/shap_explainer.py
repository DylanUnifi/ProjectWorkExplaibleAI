# explainability/shap_explainer.py

import shap
import torch
import numpy as np
from tqdm import tqdm

class SHAPExplainer:
    """
    SHAP (SHapley Additive exPlanations) for deep learning models.
    
    Uses DeepExplainer (fast) or KernelExplainer (slower but model-agnostic).
    """
    
    def __init__(self, model, background_data, method="deep"):
        """
        Args:
            model: PyTorch model
            background_data: Reference dataset for SHAP (100-1000 samples)
            method: "deep" (DeepExplainer) or "kernel" (KernelExplainer)
        """
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device
        self.method = method
        
        # Prepare background
        if isinstance(background_data, torch.utils.data.DataLoader):
            backgrounds = []
            for batch in background_data:
                if isinstance(batch, dict):
                    x = batch["image"] if "image" in batch else batch["video"]
                else:
                    x, _ = batch
                backgrounds.append(x)
                if len(backgrounds) * x.size(0) >= 100:
                    break
            self.background = torch.cat(backgrounds, dim=0)[:100].to(self.device)
        else:
            self.background = background_data.to(self.device)
        
        # Create explainer
        if method == "deep":
            self.explainer = shap.DeepExplainer(model, self.background)
        elif method == "kernel":
            def model_fn(x):
                x_torch = torch.from_numpy(x).float().to(self.device)
                with torch.no_grad():
                    out = model(x_torch)
                    if isinstance(out, tuple):
                        out = out[0]  # Take first output
                return out.cpu().numpy()
            
            self.explainer = shap.KernelExplainer(
                model_fn,
                self.background.cpu().numpy()
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def explain(self, x, target_class=None):
        """
        Compute SHAP values for input x.
        
        Args:
            x: Input tensor (B, C, H, W) or (B, C, T, H, W)
            target_class: Target class index (if None, use predicted class)
        
        Returns:
            shap_values: SHAP attributions (same shape as x)
        """
        x = x.to(self.device)
        
        if self.method == "deep":
            # IMPORTANT: DeepExplainer needs gradients; do NOT wrap in torch.no_grad()
            shap_values = self.explainer.shap_values(x, check_additivity=False)

            if isinstance(shap_values, list):
                if target_class is None:
                    with torch.no_grad():
                        preds = self.model(x)
                        if isinstance(preds, tuple):
                            preds = preds[0]
                        target_class = preds.argmax(dim=1).cpu().numpy()

                tc_list = [int(tc) for tc in (target_class if hasattr(target_class, '__iter__') else [target_class])]
                shap_vals = np.array([shap_values[tc][i] for i, tc in enumerate(tc_list)])
            else:
                shap_vals = shap_values
                # Handle ndarray with trailing class dimension, e.g. (B, C, H, W, n_classes)
                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == x.ndim + 1:
                    if target_class is None:
                        with torch.no_grad():
                            preds = self.model(x)
                            if isinstance(preds, tuple):
                                preds = preds[0]
                            target_class = preds.argmax(dim=1).cpu().numpy()
                    tc_list = [int(tc) for tc in (target_class if hasattr(target_class, '__iter__') else [target_class])]
                    shap_vals = np.stack([shap_vals[i, ..., tc] for i, tc in enumerate(tc_list)])

            return torch.from_numpy(np.ascontiguousarray(shap_vals).astype(np.float32)).to(self.device)
        
        elif self.method == "kernel":
            with torch.no_grad():
                x_np = x.detach().cpu().numpy()
            shap_values = self.explainer.shap_values(x_np, nsamples=100)
            
            if isinstance(shap_values, list):
                if target_class is None:
                    preds = self.model(x)
                    if isinstance(preds, tuple):
                        preds = preds[0]
                    target_class = preds.argmax(dim=1).cpu().numpy()

                tc_list = [int(tc) for tc in (target_class if hasattr(target_class, '__iter__') else [target_class])]
                shap_vals = np.array([shap_values[tc][i] for i, tc in enumerate(tc_list)])
            else:
                shap_vals = shap_values
                # Handle ndarray with trailing class dimension, e.g. (B, C, H, W, n_classes)
                if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == x.ndim + 1:
                    if target_class is None:
                        preds = self.model(x)
                        if isinstance(preds, tuple):
                            preds = preds[0]
                        target_class = preds.argmax(dim=1).cpu().numpy()
                    tc_list = [int(tc) for tc in (target_class if hasattr(target_class, '__iter__') else [target_class])]
                    shap_vals = np.stack([shap_vals[i, ..., tc] for i, tc in enumerate(tc_list)])

            return torch.from_numpy(np.ascontiguousarray(shap_vals).astype(np.float32)).to(self.device)

        # Should not reach here due to __init__ validation
        raise ValueError(f"Unknown method: {self.method}")
    
    def explain_batch(self, dataloader, max_batches=10):
        """Explain multiple batches."""
        all_shap = []
        all_inputs = []
        all_labels = []
        
        for i, batch in enumerate(tqdm(dataloader, desc="SHAP explain", total=max_batches)):
            if i >= max_batches:
                break
            
            if isinstance(batch, dict):
                x = batch["image"] if "image" in batch else batch["video"]
                y = batch["label"] if "label" in batch else batch["action"]
            else:
                x, y = batch
            
            shap_vals = self.explain(x)
            
            all_shap.append(shap_vals)
            all_inputs.append(x.cpu())
            all_labels.append(y)
        
        return {
            "shap_values": torch.cat(all_shap, dim=0),
            "inputs": torch.cat(all_inputs, dim=0),
            "labels": torch.cat(all_labels, dim=0),
        }
