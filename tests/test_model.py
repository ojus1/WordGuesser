import sys

sys.path.append(".")
from core.model import Net, loss_fn
import torch


def test_shapes():
    inp = torch.randint(0, 100, (32, 128))
    model = Net(100, 0, embed_dim=64)

    out = model(inp)
    out = out.argmax(dim=-1)
    assert out.shape == inp.shape


def test_loss():
    inp = torch.LongTensor([0, 1, 0, 2, 0])
    inp = torch.stack([inp] * 32, dim=0)
    pred = torch.FloatTensor(
        [
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ]
    )
    pred = torch.stack([pred] * 32, dim=0)
    l1 = loss_fn(pred, inp, padding_idx=0)

    l2 = loss_fn(pred, inp, padding_idx=1)

    l3 = loss_fn(pred, inp, padding_idx=2)
    assert l2 > l1
    assert l3 > l1
