"""
model.py — Multi-Layer Perceptron for regression (predicts customer Rating).
"""
import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Flexible MLP regressor.
    
    Architecture:  Input -> [Linear -> BatchNorm -> ReLU -> Dropout] × n_hidden -> Linear (output)
    """
    def __init__(self, input_dim: int, hidden_sizes: list = None, dropout: float = 0.2):
        super().__init__()
        if hidden_sizes is None:
            hidden_sizes = [128, 64]

        layers = []
        in_dim = input_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))   # scalar output

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def build_model(input_dim: int, hidden_sizes: list = None, dropout: float = 0.2) -> MLP:
    """Factory function: create and return an MLP."""
    model = MLP(input_dim, hidden_sizes=hidden_sizes, dropout=dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [model] MLP architecture: {input_dim} -> {hidden_sizes} -> 1")
    print(f"  [model] Trainable parameters: {n_params:,}")
    return model
