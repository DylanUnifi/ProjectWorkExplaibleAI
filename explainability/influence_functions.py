# explainability/influence_functions.py

import torch
import numpy as np

class QuantumKernelInfluence:
    """
    Influence functions pour Quantum Kernel SVM.
    Identifie quels exemples d'entraînement influencent prédictions.
    """
    
    def __init__(self, svm_model, K_train, y_train):
        self.svm = svm_model
        self.K = K_train
        self.y = y_train
    
    def compute_influence(self, test_idx, train_idx):
        """
        Influence d'un exemple train sur une prédiction test.
        
        High positive influence = removal would decrease prediction
        High negative influence = removal would increase prediction
        """
        # Get dual coefficients (support vectors)
        alpha = self.svm.dual_coef_[0]  # (n_support_vectors,)
        support_indices = self.svm.support_
        
        # Hessian approximation (kernel gram matrix)
        # H ≈ K + λI
        lambda_reg = 1.0 / self.svm.C
        H = self.K[support_indices][:, support_indices] + lambda_reg * np.eye(len(support_indices))
        
        # Inverse Hessian
        H_inv = np.linalg.inv(H)
        
        # Influence = -∇_θ L_test * H^{-1} * ∇_θ L_train
        # Simplified for kernel SVM
        
        # Gradient wrt test point
        grad_test = self.K[test_idx, support_indices]
        
        # Gradient wrt train point
        grad_train = self.K[train_idx, support_indices]
        
        # Influence score
        influence = -np.dot(grad_test @ H_inv, grad_train)
        
        return influence
    
    def find_influential_examples(self, test_idx, top_k=10):
        """
        Find top-k most influential training examples for test prediction.
        """
        n_train = len(self.y)
        influences = []
        
        for train_idx in range(n_train):
            inf = self.compute_influence(test_idx, train_idx)
            influences.append((train_idx, inf))
        
        # Sort by absolute influence
        influences.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return influences[:top_k]
