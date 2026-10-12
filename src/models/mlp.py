import torch
import torch.nn as nn
from src.models.nn_train import NNWrapper


class _MLPNet(nn.Module):
    def __init__(self, n_num, cards, width, depth, dropout):
        super().__init__()
        self.embs = nn.ModuleList(
            [nn.Embedding(c, min(32, max(2, int(round(1.6 * c ** 0.56))))) for c in cards])
        d = n_num + sum(e.embedding_dim for e in self.embs)
        layers = []
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU(), nn.Dropout(dropout)]
            d = width
        layers.append(nn.Linear(d, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, xn, xc):
        parts = []
        if xn is not None:
            parts.append(xn)
        if xc is not None:
            parts += [e(xc[:, j]) for j, e in enumerate(self.embs)]
        return self.mlp(torch.cat(parts, 1))


class MLPModel(NNWrapper):
    def __init__(self, params, task, seed, meta, cfg):
        super().__init__(params, task, seed, meta, cfg, "mlp")

    def build(self):
        p = self.p
        self.net = _MLPNet(self.meta["n_num"], self.meta["cardinalities"],
                           p["width"], p["depth"], p["dropout"])