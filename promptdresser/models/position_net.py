import torch.nn as nn

_ACTS = {"silu": nn.SiLU, "gelu": nn.GELU, "relu": nn.ReLU, "tanh": nn.Tanh}

class PositionNet(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=None, act_fn="silu"):
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        Act = _ACTS.get(act_fn, nn.SiLU)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            Act(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)
