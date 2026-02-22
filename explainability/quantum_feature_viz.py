# explainability/quantum_feature_viz.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

class QuantumFeatureSpaceVisualizer:
    """
    Visualise le feature space quantique.
    Compare avec feature space classique.
    """
    
    def __init__(self, quantum_kernel, classical_kernel=None):
        self.K_quantum = quantum_kernel
        self.K_classical = classical_kernel
    
    def visualize_kernel_pca(self, labels, save_path="kernel_pca.png"):
        """Kernel PCA pour visualiser structure."""
        from sklearn.decomposition import KernelPCA
        
        # Quantum kernel PCA
        kpca = KernelPCA(n_components=2, kernel="precomputed")
        X_quantum_2d = kpca.fit_transform(self.K_quantum)
        
        # Plot
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(
            X_quantum_2d[:, 0],
            X_quantum_2d[:, 1],
            c=labels,
            cmap="viridis",
            alpha=0.6
        )
        plt.colorbar(scatter)
        plt.title("Quantum Kernel PCA")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        
        # Classical comparison
        if self.K_classical is not None:
            X_classical_2d = kpca.fit_transform(self.K_classical)
            
            plt.subplot(1, 2, 2)
            scatter = plt.scatter(
                X_classical_2d[:, 0],
                X_classical_2d[:, 1],
                c=labels,
                cmap="viridis",
                alpha=0.6
            )
            plt.colorbar(scatter)
            plt.title("Classical Kernel PCA")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
    
    def kernel_alignment(self):
        """
        Compute alignment between quantum and classical kernels.
        High alignment = similar feature spaces.
        """
        if self.K_classical is None:
            return None
        
        # Frobenius inner product
        alignment = np.trace(self.K_quantum @ self.K_classical)
        
        # Normalize
        norm_q = np.sqrt(np.trace(self.K_quantum @ self.K_quantum))
        norm_c = np.sqrt(np.trace(self.K_classical @ self.K_classical))
        
        alignment_normalized = alignment / (norm_q * norm_c)
        
        return alignment_normalized
    
    def visualize_kernel_heatmap(self, save_path="kernel_heatmap.png"):
        """Heatmap pour voir structure du kernel."""
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.imshow(self.K_quantum, cmap="coolwarm", aspect="auto")
        plt.colorbar()
        plt.title("Quantum Kernel Matrix")
        
        if self.K_classical is not None:
            plt.subplot(1, 2, 2)
            plt.imshow(self.K_classical, cmap="coolwarm", aspect="auto")
            plt.colorbar()
            plt.title("Classical Kernel Matrix")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
