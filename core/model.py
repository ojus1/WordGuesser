import torch
from torch import nn
from .config import cfg
import torch.nn.functional as F


class GRUNet(nn.Module):
    def __init__(self, num_tokens, padding_idx, embed_dim=cfg.MODEL.embed_dim):
        super(GRUNet, self).__init__()
        self.embed = nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=embed_dim, padding_idx=padding_idx,
        )
        self.embed_dim = embed_dim
        self.n_layers = cfg.MODEL.num_layers

        self.gru = nn.GRU(
            embed_dim,
            embed_dim,
            self.n_layers,
            batch_first=True,
            dropout=cfg.MODEL.drop_out,
        )
        self.fc = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(num_tokens, num_tokens),
            nn.ReLU(inplace=True),
            nn.Linear(num_tokens, num_tokens),
        )

    def forward(self, x):
        x = self.embed(x)
        h = self.init_hidden(x.shape[0], device=x.device)
        out, h = self.gru(x, h)
        # reuse the embedding to project back to character space. Optional, doing this for parameter efficiency.
        out = out @ self.embed.weight.t()
        out = self.fc(out)
        return out

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = (
            weight.new(self.n_layers, batch_size, self.embed_dim).zero_().to(device)
        )
        return hidden


def loss_fn(pred, label, padding_idx):
    return F.cross_entropy(pred.transpose(1, 2), label, ignore_index=padding_idx)
