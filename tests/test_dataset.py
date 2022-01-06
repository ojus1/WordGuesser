import sys

sys.path.append(".")
from core.tokenizer import CharLevelTokenizer
from core.config import cfg
from core.WordDataModule import WordDataset, WordDataModule

import torch
import random

torch.manual_seed(123)
random.seed(123)


def test_ds():
    tokenizer = torch.load("./data/tokenizer.pth")
    ds = WordDataset(tokenizer, "./data/train.csv")
    ds.set_mode("train")
    tokens, mask, label = ds[2]
    assert isinstance(tokens, torch.LongTensor)
    assert isinstance(mask, torch.LongTensor)
    assert isinstance(label, torch.LongTensor)
    assert len(tokens) == len(mask) == len(label)


def test_dm():
    tokenizer = torch.load("./data/tokenizer.pth")
    dm = WordDataModule(tokenizer, "./data/train.csv")
    train_dl = dm.train_dataloader()
    for seq, mask, label in train_dl:
        assert seq.shape[1] == cfg.TOKENIZATION.max_seq_length
        assert mask.shape[1] == cfg.TOKENIZATION.max_seq_length
        assert label.shape[1] == cfg.TOKENIZATION.max_seq_length
        assert seq.shape == mask.shape == label.shape
    val_dl = dm.train_dataloader()
    for seq, mask, label in val_dl:
        assert seq.shape[1] == cfg.TOKENIZATION.max_seq_length
        assert mask.shape[1] == cfg.TOKENIZATION.max_seq_length
        assert label.shape[1] == cfg.TOKENIZATION.max_seq_length
        assert seq.shape == mask.shape == label.shape

