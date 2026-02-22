# models/clevr_qcnn.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import pennylane as qml
import numpy as np


class ResidualBlock(nn.Module):
    """Residual block for 2D CNN (same as HybridQCNNBinaryClassifier)."""
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        stride = 2 if downsample else 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential()
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


def _create_quantum_layer(n_qubits, n_layers=2, backend="lightning.qubit"):
    """Create PennyLane quantum layer."""
    dev = qml.device(backend, wires=n_qubits)

    @qml.qnode(dev, interface="torch")
    def qnode(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Z")
        qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    for name, param in layer.named_parameters():
        if "weights" in name:
            nn.init.uniform_(param, -np.pi / 2, np.pi / 2)

    return layer


class CLEVRQCNNClassifier(nn.Module):
    """
    QCNN for CLEVR-Hans (3 or 7 classes).
    Adapted from HybridQCNNBinaryClassifier for multi-class classification.
    """

    def __init__(
        self,
        n_classes=3,
        input_channel=3,
        n_qubits=4,
        n_layers=1,
        backend="lightning.qubit",
        conv_channels=None,
        hidden_sizes=None,
        dropout=0.3,
    ):
        super().__init__()
        self.input_channel = input_channel

        if conv_channels is None:
            conv_channels = [32, 64, 128]

        # 2D Conv backbone
        self.conv_blocks = nn.ModuleList()
        in_ch = input_channel
        for idx, out_ch in enumerate(conv_channels):
            downsample = idx > 0
            self.conv_blocks.append(ResidualBlock(in_ch, out_ch, downsample=downsample))
            in_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout)

        # Classical head
        if hidden_sizes is None:
            hidden_sizes = []

        fc_layers = []
        prev_dim = in_ch
        for hidden_dim in hidden_sizes:
            fc_layers.append(nn.Linear(prev_dim, hidden_dim))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.classical_head = nn.Sequential(*fc_layers)

        # Quantum layer
        self.quantum_fc_input = nn.Linear(prev_dim, n_qubits)
        self.quantum_layer = _create_quantum_layer(n_qubits, n_layers, backend=backend)
        self.bn_q = nn.LayerNorm(n_qubits)

        # Multi-class output
        self.final_fc = nn.Linear(n_qubits, n_classes)

    def forward(self, x, return_features=False):
        target_device = x.device

        for block in self.conv_blocks:
            x = block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classical_head(x)

        # Quantum layer with residual connection
        # Project to quantum dimensions
        x_proj = self.quantum_fc_input(x)
        x_q_input = torch.tanh(x_proj) * np.pi

        batch_size = x_q_input.shape[0]
        n_q = self.quantum_fc_input.out_features
        x_quantum = self.quantum_layer(x_q_input)
        x_quantum = x_quantum.reshape(batch_size, n_q)
        x_quantum = x_quantum.to(target_device)

        # Residual: classical projection + quantum output
        x_combined = x_proj + x_quantum
        x_combined = self.bn_q(x_combined)

        # Multi-class logits
        logits = self.final_fc(x_combined)

        if return_features:
            return logits, x_combined

        return logits
