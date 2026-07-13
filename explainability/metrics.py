# explainability/metrics.py

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.stats import spearmanr

class XAIMetrics:
    """
    Metrics to evaluate explanation quality.
    
    Implemented metrics:
    - Faithfulness (Insertion/Deletion)
    - Robustness (Stability under perturbations)
    - Sensitivity (Attribution consistency)
    - Infidelity
    - Sparsity
    """
    
    def __init__(self, model, device="cuda"):
        self.model = model
        self.model.eval()
        self.device = device
    
    def _sanitize_inputs(self, x, attribution, target):
        """
        Canonicalize inputs for metric functions.

        - x: float32 tensor on self.device, shape (B, C, H, W) or (B, C, T, H, W)
        - attribution: float32 tensor on self.device, shape matching x
          Handles:
            BxHxW -> Bx1xHxW
            Bx1xHxW with x having C>1 channels -> expand to BxCxHxW
            BxCxHxWxK (SHAP trailing class dim) -> select target class
        - target: 1-D LongTensor of shape (B,) on self.device
        """
        # Canonicalize x
        x = x.to(device=self.device, dtype=torch.float32)
        B = x.shape[0]

        # Canonicalize target
        if target is None:
            with torch.no_grad():
                out = self.model(x)
                if isinstance(out, tuple):
                    out = out[0]
                target = out.argmax(dim=1)
        if not isinstance(target, torch.Tensor):
            target = torch.tensor(target, dtype=torch.long, device=self.device)
        target = target.to(device=self.device, dtype=torch.long)
        if target.numel() == 1:
            target = target.expand(B)
        target = target.reshape(B)

        # Canonicalize attribution
        if not isinstance(attribution, torch.Tensor):
            attribution = torch.tensor(np.array(attribution), dtype=torch.float32)
        attribution = attribution.detach().to(device=self.device, dtype=torch.float32)

        # Handle SHAP trailing class dimension: (B, C, H, W, K) -> (B, C, H, W)
        if attribution.ndim == x.ndim + 1:
            tc_list = target.tolist()
            attribution = torch.stack(
                [attribution[i, ..., tc] for i, tc in enumerate(tc_list)]
            )

        # Handle missing batch dim: (C, H, W) -> (1, C, H, W)
        if attribution.ndim == x.ndim - 1:
            attribution = attribution.unsqueeze(0)

        # Handle missing channel dim: (B, H, W) -> (B, 1, H, W)
        if attribution.ndim == x.ndim - 1:
            attribution = attribution.unsqueeze(1)

        # Expand single-channel attribution to match x channels
        if attribution.ndim == x.ndim and attribution.shape[1] == 1 and x.shape[1] > 1:
            attribution = attribution.expand_as(x).contiguous()

        # Final element-count check: re-flatten/reshape if needed
        if attribution.numel() != x.numel():
            # Last resort: reshape to match x using the flattened data
            attr_flat = attribution.reshape(B, -1)
            x_flat = x.reshape(B, -1)
            n_x = x_flat.shape[1]
            n_a = attr_flat.shape[1]
            if n_a > n_x:
                attr_flat = attr_flat[:, :n_x]
            elif n_a < n_x:
                pad = torch.zeros(B, n_x - n_a, device=self.device, dtype=torch.float32)
                attr_flat = torch.cat([attr_flat, pad], dim=1)
            attribution = attr_flat.reshape_as(x)

        return x, attribution, target

    def faithfulness_insertion(self, x, attribution, steps=10, target=None):
        """
        Insertion metric: Insert pixels by importance order.
        Good explanations should quickly recover prediction.
        
        Returns:
            auc: Area under insertion curve (higher = better)
        """
        x, attribution, target = self._sanitize_inputs(x, attribution, target)
        
        # Get original prediction
        with torch.no_grad():
            orig_output = self.model(x)
            if isinstance(orig_output, tuple):
                orig_output = orig_output[0]
            
            orig_prob = F.softmax(orig_output, dim=1)[range(len(target)), target]
        
        # Flatten spatial dimensions
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        attr_flat = attribution.abs().reshape(B, -1)
        
        # Sort by importance (descending)
        sorted_indices = torch.argsort(attr_flat, dim=1, descending=True)
        
        # Start with baseline (zeros)
        x_masked = torch.zeros_like(x)
        
        # Insert pixels progressively
        probs = []
        n_pixels = x_flat.shape[1]
        step_size = n_pixels // steps
        
        for step in range(steps + 1):
            n_inserted = step * step_size
            
            # Insert top-n pixels
            x_masked_flat = x_masked.reshape(B, -1)
            for b in range(B):
                indices = sorted_indices[b, :n_inserted]
                x_masked_flat[b, indices] = x_flat[b, indices]
            x_masked = x_masked_flat.reshape_as(x)
            
            # Predict
            with torch.no_grad():
                output = self.model(x_masked)
                if isinstance(output, tuple):
                    output = output[0]
                prob = F.softmax(output, dim=1)[range(len(target)), target]
                probs.append(prob.cpu())
        
        # Compute AUC
        probs = torch.stack(probs, dim=1)  # (B, steps+1)
        auc = probs.mean(dim=1)  # Average over steps
        
        return auc.mean().item()
    
    def faithfulness_deletion(self, x, attribution, steps=10, target=None):
        """
        Deletion metric: Remove pixels by importance order.
        Good explanations should quickly decrease prediction.
        
        Returns:
            auc: Area under deletion curve (lower = better)
        """
        x, attribution, target = self._sanitize_inputs(x, attribution, target)
        
        # Get original prediction
        with torch.no_grad():
            orig_output = self.model(x)
            if isinstance(orig_output, tuple):
                orig_output = orig_output[0]
            
            orig_prob = F.softmax(orig_output, dim=1)[range(len(target)), target]
        
        # Flatten
        B = x.shape[0]
        x_flat = x.reshape(B, -1)
        attr_flat = attribution.abs().reshape(B, -1)
        
        # Sort by importance
        sorted_indices = torch.argsort(attr_flat, dim=1, descending=True)
        
        # Start with full image
        x_masked = x.clone()
        
        # Delete pixels progressively
        probs = []
        n_pixels = x_flat.shape[1]
        step_size = n_pixels // steps
        
        for step in range(steps + 1):
            n_deleted = step * step_size
            
            # Delete top-n pixels
            x_masked_flat = x_masked.reshape(B, -1)
            for b in range(B):
                indices = sorted_indices[b, :n_deleted]
                x_masked_flat[b, indices] = 0
            x_masked = x_masked_flat.reshape_as(x)
            
            # Predict
            with torch.no_grad():
                output = self.model(x_masked)
                if isinstance(output, tuple):
                    output = output[0]
                prob = F.softmax(output, dim=1)[range(len(target)), target]
                probs.append(prob.cpu())
        
        # Compute AUC (lower is better for deletion)
        probs = torch.stack(probs, dim=1)
        auc = probs.mean(dim=1)
        
        return auc.mean().item()
    
    def robustness_stability(self, x, explainer, n_perturbations=10, noise_std=0.1):
        """
        Robustness: Stability of explanations under small input perturbations.
        
        Returns:
            stability: Spearman correlation between explanations (higher = better)
        """
        x = x.to(self.device)
        
        # Get base explanation
        base_attr = explainer.explain(x)
        base_attr_flat = base_attr.view(x.shape[0], -1).detach().cpu().numpy()
        
        correlations = []
        
        for _ in range(n_perturbations):
            # Add Gaussian noise
            noise = torch.randn_like(x) * noise_std
            x_perturbed = x + noise
            x_perturbed = torch.clamp(x_perturbed, 0, 1)
            
            # Get perturbed explanation
            perturbed_attr = explainer.explain(x_perturbed)
            perturbed_attr_flat = perturbed_attr.view(x.shape[0], -1).detach().cpu().numpy()
            
            # Compute Spearman correlation per sample
            for i in range(len(x)):
                corr, _ = spearmanr(base_attr_flat[i], perturbed_attr_flat[i])
                correlations.append(corr)
        
        return np.mean(correlations)
    
    def infidelity(self, x, attribution, target=None, n_samples=100):
        """
        Infidelity: Measures how well attributions explain model predictions.
        
        Lower is better.
        """
        x, attribution, target = self._sanitize_inputs(x, attribution, target)
        
        # Original prediction
        with torch.no_grad():
            orig_output = self.model(x)
            if isinstance(orig_output, tuple):
                orig_output = orig_output[0]
            
            orig_score = orig_output[range(len(target)), target]
        
        infidelity_scores = []
        
        for _ in range(n_samples):
            # Random perturbation
            perturbation = torch.randn_like(x) * 0.1
            x_perturbed = x + perturbation
            
            # Prediction change
            with torch.no_grad():
                perturbed_output = self.model(x_perturbed)
                if isinstance(perturbed_output, tuple):
                    perturbed_output = perturbed_output[0]
                perturbed_score = perturbed_output[range(len(target)), target]
            
            output_diff = (orig_score - perturbed_score).abs()
            
            # Attribution-weighted perturbation
            attr_weighted_pert = (attribution * perturbation).view(len(x), -1).sum(dim=1).abs()
            
            # Infidelity = |output_diff - attr_weighted_pert|
            infid = (output_diff - attr_weighted_pert).abs()
            infidelity_scores.append(infid.cpu())
        
        return torch.stack(infidelity_scores).mean().item()
    
    def sparsity(self, attribution, threshold=0.1):
        """
        Sparsity: Fraction of near-zero attributions.
        Higher sparsity = more interpretable (focuses on few features).
        
        Returns:
            sparsity: Fraction of values below threshold
        """
        attr_abs = attribution.abs()
        attr_normalized = attr_abs / (attr_abs.max() + 1e-8)
        
        sparse_mask = attr_normalized < threshold
        sparsity_score = sparse_mask.float().mean().item()
        
        return sparsity_score
    
    def evaluate_all(self, x, attribution, explainer=None, target=None):
        """
        Evaluate all metrics for given attribution.
        
        Returns:
            dict of metrics
        """
        x, attribution, target = self._sanitize_inputs(x, attribution, target)
        metrics = {
            "faithfulness_insertion": self.faithfulness_insertion(x, attribution, target=target),
            "faithfulness_deletion": self.faithfulness_deletion(x, attribution, target=target),
            "infidelity": self.infidelity(x, attribution, target=target),
            "sparsity": self.sparsity(attribution),
        }
        
        if explainer is not None:
            metrics["robustness"] = self.robustness_stability(x, explainer)
        
        return metrics


# Specific metrics for confounded datasets (CLEVR-Hans)
class ConfounderDetectionMetrics:
    """
    Metrics specific to detecting whether explanations
    capture confounders vs. true class rules.
    """
    
    @staticmethod
    def confounder_attribution_ratio(attribution, confounder_mask):
        """
        Ratio of attribution on confounders vs. useful features.
        
        Args:
            attribution: (B, C, H, W) attribution map
            confounder_mask: (B, C, H, W) binary mask (1 = confounder region)
        
        Returns:
            ratio: Attribution on confounders / Total attribution
        """
        attr_abs = attribution.abs()
        
        # Attribution on confounders
        confounder_attr = (attr_abs * confounder_mask).sum()
        
        # Total attribution
        total_attr = attr_abs.sum()
        
        ratio = (confounder_attr / (total_attr + 1e-8)).item()
        
        return ratio
    
    @staticmethod
    def object_level_attribution(attribution, scene_graph, target_objects):
        """
        Compute attribution per object in CLEVR scene.
        
        Args:
            attribution: (C, H, W) attribution map
            scene_graph: List of object bounding boxes
            target_objects: List of object IDs that should be important
        
        Returns:
            precision: Fraction of important objects correctly identified
            recall: Fraction of attributed regions on target objects
        """
        # Aggregate attribution per object bbox
        object_scores = {}
        
        for obj in scene_graph:
            obj_id = obj["id"]
            bbox = obj["bbox"]  # [x, y, w, h]
            
            # Extract attribution in bbox
            x, y, w, h = bbox
            attr_region = attribution[:, y:y+h, x:x+w]
            
            # Score = sum of attribution
            object_scores[obj_id] = attr_region.abs().sum().item()
        
        # Rank objects by attribution
        ranked_objects = sorted(object_scores.items(), key=lambda x: x[1], reverse=True)
        top_k = len(target_objects)
        predicted_important = [obj_id for obj_id, _ in ranked_objects[:top_k]]
        
        # Precision & Recall
        tp = len(set(predicted_important) & set(target_objects))
        precision = tp / len(predicted_important) if predicted_important else 0
        recall = tp / len(target_objects) if target_objects else 0
        
        return precision, recall
