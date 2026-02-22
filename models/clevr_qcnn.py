# models/clevr_qcnn.py

# Réutilise ton HybridQCNNBinaryClassifier mais adapte pour multi-classe

class CLEVRQCNNClassifier(nn.Module):
    """QCNN pour CLEVR-Hans (3 ou 7 classes)."""
    
    def __init__(self, n_classes=3, **kwargs):
        super().__init__()
        
        # Utilise la même architecture que HybridQCNN
        # mais change le dernier layer
        
        # ... (copie architecture HybridQCNN) ...
        
        self.final_fc = nn.Linear(n_qubits, n_classes)  # Multi-classe
    
    def forward(self, x, return_features=False):
        # ... (même forward) ...
        
        logits = self.final_fc(x_quantum)
        
        if return_features:
            return logits, x_quantum
        
        return logits
