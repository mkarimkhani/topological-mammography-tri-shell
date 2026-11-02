import torch
import torch.nn as nn

class TriShellMaskGenerator(nn.Module):
    def __init__(self, size=224, r1=0.25, r2=0.55, r3=0.90, softness=0.04):
        super().__init__()
        self.size = size
        self.r1, self.r2, self.r3 = r1, r2, r3
        self.softness = softness
        coords = torch.linspace(-1.0, 1.0, steps=size)
        y, x = torch.meshgrid(coords, coords, indexing='ij')
        dist = torch.sqrt(x**2 + y**2)
        self.register_buffer('dist', dist.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        B = x.shape[0]
        d = self.dist.expand(B, -1, -1, -1)
        m_core = torch.sigmoid((self.r1 - d) / self.softness)
        m_margin = torch.sigmoid((d - self.r1) / self.softness) * torch.sigmoid((self.r2 - d) / self.softness)
        m_stroma = torch.sigmoid((d - self.r2) / self.softness) * torch.sigmoid((self.r3 - d) / self.softness)
        return torch.cat([m_core, m_margin, m_stroma], dim=1)

class TopologicalLatentModulator(nn.Module):
    def __init__(self, topo_dim=15, feat_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(topo_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, feat_dim * 2)
        )

    def forward(self, deep_features, topo_vector):
        film_params = self.mlp(topo_vector)
        gamma, beta = torch.chunk(film_params, 2, dim=-1)
        gamma = torch.tanh(gamma)
        return deep_features * (1.0 + gamma) + beta
