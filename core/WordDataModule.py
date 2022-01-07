import pytorch_lightning as pl
import torch
from .config import cfg
from .tokenizer import CharLevelTokenizer
import pandas as pd


class WordDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, csv_path, indices=None):
        super().__init__()
        self.raw = pd.read_csv(csv_path).dropna(axis=0)
        self.tokenizer = tokenizer
        # Use indices if given, else use whole csv
        self.indices = indices if not indices is None else list(range(len(self.raw)))

    def set_mode(self, mode):
        assert mode in ["train", "val"]
        self.mode = mode

    def __getitem__(self, idx):
        idx = self.indices[idx]
        word = self.raw.iloc[idx % len(self.raw), 0]
        description = self.raw.iloc[idx % len(self.raw), 1]
        if self.mode == "val":
            seed = idx  # Use deterministic masking when validating
        else:
            seed = None
        tokens, mask_ids, label = self.tokenizer.encode(
            word, description, mode=self.mode
        )
        tokens = torch.LongTensor(tokens)
        label = torch.LongTensor(label)

        one_hot_mask_ids = torch.zeros_like(tokens)
        one_hot_mask_ids[mask_ids] = 1
        # if label.shape[0] != 400:
        #     print(one_hot_mask_ids.shape, label.shape, self.mode)
        label = label.masked_fill(
            ~one_hot_mask_ids.bool(),
            self.tokenizer.tokens_to_ids[cfg.TOKENIZATION.pad_token],
        )
        return tokens, one_hot_mask_ids, label

    def __len__(self):
        if self.mode == "val":
            # when validating, use different masks of the same example to get a better estimate of performance
            return len(self.indices) * cfg.TRAIN.repeat_val_dataset
        return len(self.indices)


class WordDataModule(pl.LightningDataModule):
    def __init__(self, tokenizer, csv_path):
        super().__init__()
        tmp = pd.read_csv(csv_path).dropna(axis=0)
        all_idx = list(range(len(tmp)))

        from sklearn.model_selection import train_test_split

        train_idx, val_idx = train_test_split(all_idx, train_size=cfg.TRAIN.train_size)

        self.train = WordDataset(tokenizer, csv_path, indices=train_idx)
        self.train.set_mode("train")
        self.val = WordDataset(tokenizer, csv_path, indices=val_idx)
        self.val.set_mode("val")

    def train_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.train,
            batch_size=cfg.TRAIN.batch_size,
            shuffle=True,
            num_workers=cfg.TRAIN.num_workers,
        )
        return dl

    def val_dataloader(self):
        dl = torch.utils.data.DataLoader(
            self.val,
            batch_size=cfg.TRAIN.batch_size,
            shuffle=False,
            num_workers=cfg.TRAIN.num_workers,
        )
        return dl
