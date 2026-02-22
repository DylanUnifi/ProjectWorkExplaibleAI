# explainability/kernel_shap_qsvm.py

import shap
import numpy as np

class QuantumKernelSHAP:
    """
    SHAP pour Quantum Kernel SVM.
    Explique en termes de features (post-PCA).
    """
    
    def __init__(self, svm_model, kernel_matrix, X_train):
        self.svm = svm_model
        self.K = kernel_matrix
        self.X_train = X_train
    
    def explain(self, X_test, background_size=100):
        """
        Explain predictions using Kernel SHAP.
        
        Returns:
            shap_values: Attribution sur features PCA (not pixels!)
        """
        # Background data
        background = shap.sample(self.X_train, background_size)
        
        # Define prediction function
        def predict_fn(X):
            # Compute kernel between X and train data
            K_test = compute_kernel_matrix(
                X, Y=self.X_train,
                weights=self.quantum_weights,
                symmetric=False,
            )
            return self.svm.decision_function(K_test)
        
        # Kernel SHAP explainer
        explainer = shap.KernelExplainer(predict_fn, background)
        
        # Compute SHAP values
        shap_values = explainer.shap_values(X_test, nsamples=100)
        
        return shap_values
    
    def visualize_feature_importance(self, shap_values, feature_names=None):
        """Bar plot of feature importance."""
        import matplotlib.pyplot as plt
        
        # Average absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        if feature_names is None:
            feature_names = [f"PC{i+1}" for i in range(len(importance))]
        
        # Sort
        indices = np.argsort(importance)[::-1][:20]  # Top 20
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel("Mean |SHAP value|")
        plt.title("Quantum Feature Importance")
        plt.tight_layout()
        plt.savefig("quantum_kernel_feature_importance.png")
