import torch, torch.nn as nn, os
from torchview import draw_graph

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 64, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 64, 3, 1, 1)
    def forward(self, x):
        return self.conv2(self.relu1(self.conv1(x)))

class Quantize(nn.Module):
    """Codebook lookup: distance computation → argmin → embedding lookup"""
    def __init__(self):
        super().__init__()
        self.codebook = nn.Embedding(16, 64)
    def forward(self, z_e):
        f = z_e.permute(0,2,1).reshape(-1, 64)
        d = (f.pow(2).sum(1,keepdim=True) - 2*f@self.codebook.weight.T
             + self.codebook.weight.pow(2).sum(1,keepdim=True).T)
        return self.codebook(d.argmin(1)).view_as(z_e)

class QuantizerSTE(nn.Module):
    """Wraps Quantize with straight-through estimator bypass"""
    def __init__(self):
        super().__init__()
        self.quantize = Quantize()
    def forward(self, z_e):
        z_q = self.quantize(z_e)
        return z_e + (z_q - z_e).detach()

class QuantizerNoSTE(nn.Module):
    """Wraps Quantize without any gradient bypass"""
    def __init__(self):
        super().__init__()
        self.quantize = Quantize()
    def forward(self, z_e):
        return self.quantize(z_e)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(64, 64, 3, 1, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(64, 1, 3, 1, 1)
    def forward(self, x):
        return self.conv2(self.relu1(self.conv1(x)))

class VQVAESte(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.quantizer = QuantizerSTE()
        self.decoder = Decoder()
    def forward(self, x):
        return self.decoder(self.quantizer(self.encoder(x)))

class VQVAENoSte(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.quantizer = QuantizerNoSTE()
        self.decoder = Decoder()
    def forward(self, x):
        return self.decoder(self.quantizer(self.encoder(x)))

device, x = torch.device("cpu"), torch.randn(1, 1, 64)
out = os.path.abspath("images")

for name, model in [("vqvae_with_ste", VQVAESte()), ("vqvae_no_ste", VQVAENoSte())]:
    g = draw_graph(model.eval(), input_data=x, save_graph=False, depth=2, graph_name=name, device=device)
    dot_path = os.path.join(out, f"{name}.gv")
    g.visual_graph.save(dot_path)
    print(f"Saved DOT: {name}.gv")
