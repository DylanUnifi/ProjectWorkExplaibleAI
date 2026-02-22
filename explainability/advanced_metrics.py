# explainability/advanced_metrics.py

"""
Advanced XAI evaluation metrics for comparative study.
"""

import torch
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import average_precision_score

class ComparativeXAIMetrics:
    """
    Métriques pour comparer différentes méthodes XAI:
    - Agreement entre méthodes
    - Stability across models
    - Confounder sensitivity
    """
    
    @staticmethod
    def rank_agreement(attr1, attr2, method="spearman"):
        """
        Measure rank correlation between two attribution maps.
        High correlation = methods agree on important features.
        
        Args:
            attr1, attr2: (B, C, H, W) attribution tensors
            method: "spearman" or "kendall"
        
        Returns:
            correlation: Rank correlation coefficient
        """
        # Flatten
        attr1_flat = attr1.abs().view(attr1.shape[0], -1).cpu().numpy()
        attr2_flat = attr2.abs().view(attr2.shape[0], -1).cpu().numpy()
        
        correlations = []
        
        for i in range(len(attr1_flat)):
            if method == "spearman":
                corr, _ = spearmanr(attr1_flat[i], attr2_flat[i])
            elif method == "kendall":
                corr, _ = kendalltau(attr1_flat[i], attr2_flat[i])
            else:
                raise ValueError(f"Unknown method: {method}")
            
            correlations.append(corr)
        
        return np.mean(correlations)
    
    @staticmethod
    def top_k_overlap(attr1, attr2, k=100):
        """
        Measure overlap of top-k most important pixels.
        
        Returns:
            overlap: Fraction of top-k pixels that overlap
        """
        attr1_flat = attr1.abs().view(attr1.shape[0], -1)
        attr2_flat = attr2.abs().view(attr2.shape[0], -1)
        
        overlaps = []
        
        for i in range(len(attr1_flat)):
            # Get top-k indices
            _, top_k_1 = torch.topk(attr1_flat[i], k)
            _, top_k_2 = torch.topk(attr2_flat[i], k)
            
            # Compute overlap
            intersection = len(set(top_k_1.tolist()) & set(top_k_2.tolist()))
            overlap = intersection / k
            
            overlaps.append(overlap)
        
        return np.mean(overlaps)
    
    @staticmethod
    def confounder_sensitivity(attribution, confounder_mask):
        """
        Measure how much attribution focuses on confounders vs. true features.
        
        Args:
            attribution: (B, C, H, W) attribution map
            confounder_mask: (B, H, W) binary mask (1 = confounder region)
        
        Returns:
            sensitivity: Ratio of attribution on confounders
        """
        attr_abs = attribution.abs().sum(dim=1)  # (B, H, W)
        
        # Upsample mask if needed
        if confounder_mask.shape != attr_abs.shape:
            confounder_mask = torch.nn.functional.interpolate(
                confounder_mask.unsqueeze(1).float(),
                size=attr_abs.shape[1:],
                mode="nearest"
            ).squeeze(1)
        
        # Attribution on confounders
        confounder_attr = (attr_abs * confounder_mask).sum(dim=(1, 2))
        
        # Total attribution
        total_attr = attr_abs.sum(dim=(1, 2))
        
        # Ratio
        sensitivity = (confounder_attr / (total_attr + 1e-8)).mean().item()
        
        return sensitivity
    
    @staticmethod
    def localization_accuracy(attribution, ground_truth_mask):
        """
        Measure how well attribution localizes important regions.
        
        Uses Average Precision (AP) metric.
        
        Args:
            attribution: (B, C, H, W) attribution map
            ground_truth_mask: (B, H, W) binary mask (1 = important)
        
        Returns:
            ap: Average Precision score
        """
        attr_abs = attribution.abs().sum(dim=1)  # (B, H, W)
        
        # Flatten
        attr_flat = attr_abs.view(attr_abs.shape[0], -1).cpu().numpy()
        mask_flat = ground_truth_mask.view(ground_truth_mask.shape[0], -1).cpu().numpy()
        
        # Compute AP per sample
        aps = []
        for i in range(len(attr_flat)):
            try:
                ap = average_precision_score(mask_flat[i], attr_flat[i])
                aps.append(ap)
            except:
                pass
        
        return np.mean(aps) if aps else 0.0
    
    @staticmethod
    def explanation_complexity(attribution, threshold=0.1):
        """
        Measure complexity of explanation.
        Lower complexity = more interpretable (fewer important features).
        
        Returns:
            complexity: Number of "active" pixels (above threshold)
        """
        attr_abs = attribution.abs()
        attr_normalized = attr_abs / (attr_abs.max() + 1e-8)
        
        # Count active pixels
        active_mask = (attr_normalized > threshold).float()
        complexity = active_mask.mean().item()
        
        return complexity
    
    @staticmethod
    def compare_all_methods(explanations_dict, metrics_to_compute=None):
        """
        Compare multiple explanation methods.
        
        Args:
            explanations_dict: Dict[method_name] = attribution_tensor
            metrics_to_compute: List of metric names
        
        Returns:
            comparison_matrix: Dict of pairwise comparisons
        """
        if metrics_to_compute is None:
            metrics_to_compute = ["rank_agreement", "top_k_overlap"]
        
        methods = list(explanations_dict.keys())
        n_methods = len(methods)
        
        comparison = {}
        
        for metric_name in metrics_to_compute:
            comparison[metric_name] = np.zeros((n_methods, n_methods))
            
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods):
                    if i == j:
                        comparison[metric_name][i, j] = 1.0
                        continue
                    
                    attr1 = explanations_dict[method1]
                    attr2 = explanations_dict[method2]
                    
                    if metric_name == "rank_agreement":
                        score = ComparativeXAIMetrics.rank_agreement(attr1, attr2)
                    elif metric_name == "top_k_overlap":
                        score = ComparativeXAIMetrics.top_k_overlap(attr1, attr2, k=100)
                    else:
                        score = 0.0
                    
                    comparison[metric_name][i, j] = score
        
        return comparison, methods
