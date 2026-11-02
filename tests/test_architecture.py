import torch
from topological_latent_modulation import TriShellMaskGenerator, TopologicalLatentModulator

def test_forward():
    B = 2
    x = torch.randn(B, 3, 224, 224)
    masks = TriShellMaskGenerator(size=224)(x)
    assert masks.shape == (B, 3, 224, 224)
    mod = TopologicalLatentModulator(topo_dim=15, feat_dim=512)
    out = mod(torch.randn(B, 512), torch.randn(B, 15))
    assert out.shape == (B, 512)
    print("All architecture unit tests passed successfully!")

if __name__ == "__main__":
    test_forward()
