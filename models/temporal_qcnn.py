# models/temporal_qcnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np

class TemporalResidualBlock(nn.Module):
    """Residual block for 3D CNN (spatiotemporal)."""
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = (1, 2, 2) if downsample else (1, 1, 1)  # Don't downsample time too much
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class TemporalQCNN(nn.Module):
    """
    3D CNN + Quantum layer for video action prediction (BDD-OIA).
    
    Input: (B, C, T, H, W) - video clips
    Output: (B, 4) - action logits + (B, 21) - explanation logits
    """
    
    def __init__(
        self,
        in_channels=3,
        n_qubits=8,
        n_layers=2,
        backend="lightning.gpu",
        conv_channels=[32, 64, 128],
        hidden_sizes=[512, 256],
        n_actions=4,
        n_explanations=21,
        dropout=0.3,
    ):
        super().__init__()
        
        # 3D Conv backbone
        self.conv_blocks = nn.ModuleList()
        prev_ch = in_channels
        for idx, out_ch in enumerate(conv_channels):
            downsample = idx > 0
            self.conv_blocks.append(
                TemporalResidualBlock(prev_ch, out_ch, downsample=downsample)
            )
            prev_ch = out_ch
        
        # Temporal pooling
        self.temporal_pool = nn.AdaptiveAvgPool3d((1, 1, 1))  # Pool spatiotemporal
        
        # Classical head
        fc_layers = []
        prev_dim = prev_ch
        for hidden_dim in hidden_sizes:
            fc_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        self.classical_head = nn.Sequential(*fc_layers)
        
        # Quantum layer
        self.quantum_fc = nn.Linear(prev_dim, n_qubits)
        self.quantum_layer = self._create_quantum_layer(n_qubits, n_layers, backend)
        self.bn_quantum = nn.LayerNorm(n_qubits)
        
        # Multi-task heads
        self.action_head = nn.Linear(n_qubits, n_actions)  # Action classification
        self.explanation_head = nn.Linear(n_qubits, n_explanations)  # Explanation (multi-label)
    
    def _create_quantum_layer(self, n_qubits, n_layers, backend):
        """Create quantum layer."""
        dev = qml.device(backend, wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def qnode(inputs, weights):
            qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))
        
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        return qml.qnn.TorchLayer(qnode, weight_shapes)
    
    def forward(self, x, return_features=False):
        """
        Args:
            x: (B, C, T, H, W) video tensor
        
        Returns:
            action_logits: (B, 4)
            explanation_logits: (B, 21)
            features: (B, n_qubits) [if return_features=True]
        """
        # 3D Conv
        for block in self.conv_blocks:
            x = block(x)
        
        # Pool
        x = self.temporal_pool(x)  # (B, C, 1, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        
        # Classical head
        x = self.classical_head(x)  # (B, hidden)
        
        # Quantum layer
        x_quantum_input = torch.tanh(self.quantum_fc(x)) * np.pi
        batch_size = x_quantum_input.shape[0]
        x_quantum = self.quantum_layer(x_quantum_input)
        # PennyLane v0.44+ may return flat tensor; reshape to (batch, n_qubits)
        if x_quantum.dim() == 1:
            x_quantum = x_quantum.unsqueeze(0)
        if x_quantum.shape[0] != batch_size:
            n_qubits_out = x_quantum.numel() // batch_size
            x_quantum = x_quantum.reshape(batch_size, n_qubits_out)
        x_quantum = self.bn_quantum(x_quantum)
        
        # Multi-task outputs
        action_logits = self.action_head(x_quantum)
        explanation_logits = self.explanation_head(x_quantum)
        
        if return_features:
            return action_logits, explanation_logits, x_quantum
        
        return action_logits, explanation_logits
