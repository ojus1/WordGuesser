from .config import cfg
import random
import torch


class CharLevelTokenizer:
    def __init__(self):
        self.tokens = [
            cfg.TOKENIZATION.mask_token,
            cfg.TOKENIZATION.sep_token,
            cfg.TOKENIZATION.cls_token,
            cfg.TOKENIZATION.pad_token,
            cfg.TOKENIZATION.unknown_token,
        ]

        self.update_index()

    def update_index(self):
        self.tokens_to_ids = {k: v for v, k in enumerate(self.tokens)}
        self.ids_to_tokens = {v: k for v, k in enumerate(self.tokens)}

    def add_tokens(self, tokens):
        for tk in tokens:
            assert tk not in self.tokens
            self.tokens.append(tk)
        self.update_index()

    def mask_word(self, word: str, mode: str, seed=None):
        word = word.lower()
        num_letters = len(word)
        if not seed is None:
            torch.manual_seed(seed)
            random.seed(seed)
        mask_ratio = torch.FloatTensor([0]).uniform_(
            cfg.TRAIN.min_mask_ratio, cfg.TRAIN.max_mask_ratio
        )
        # Mask at least one letter
        num_letters_to_mask = max(int(mask_ratio * num_letters), 1)
        mask_idx = torch.LongTensor(
            random.sample(list(range(num_letters)), k=num_letters_to_mask)
        )
        not_masked_idx = torch.LongTensor(
            [
                i
                for i in range(num_letters)
                if i not in mask_idx and random.random() > cfg.TRAIN.replace_prob
            ]
        )

        new_word = []
        gt = []
        for i, s in enumerate(word):
            if i not in mask_idx:
                new_word.append(s.lower())
            else:
                new_word.append(cfg.TOKENIZATION.mask_token)
            gt.append(s)
        if mode == "train":
            mask_idx = torch.concat(
                [mask_idx, not_masked_idx]
            )  # some letters will not be masked but be included in the loss, see BERT paper for details.
        return new_word, mask_idx, gt

    def encode(self, word: str, description: str, mode: str, seed=None):
        if mode != "test":
            word, mask_idx, gt = self.mask_word(word, seed=seed, mode=mode)
        else:
            word, mask_idx = self.clean_test_word(word)
            gt = None

        mask_idx = mask_idx + 1  # Offset by 1 since [CLS] is appended at the start

        description = [tk for tk in description]
        example = (
            [cfg.TOKENIZATION.cls_token]
            + word
            + [cfg.TOKENIZATION.sep_token]
            + description
        )
        example, gt = self.pad(example, gt)
        if mode != "test":
            gt = [self.encode_one_token(tk) for tk in gt]
        example = example + [cfg.TOKENIZATION.sep_token]
        return [self.encode_one_token(tk) for tk in example], mask_idx, gt

    def pad(self, ex, label=None):
        # pad if too short, or delete tokens from the end
        if len(ex) > cfg.TOKENIZATION.max_seq_length - 1:  # Save one token for [SEP]
            ex = ex[: cfg.TOKENIZATION.max_seq_length - 1]
        else:
            to_pad = abs(len(ex) - (cfg.TOKENIZATION.max_seq_length - 1))
            ex = ex + [cfg.TOKENIZATION.pad_token] * to_pad
        if not label is None:
            if len(label) > cfg.TOKENIZATION.max_seq_length:
                label = label[: cfg.TOKENIZATION.max_seq_length]
            else:
                to_pad = abs(len(label) - cfg.TOKENIZATION.max_seq_length)
                label = label + [cfg.TOKENIZATION.pad_token] * to_pad
            assert len(label) == len(ex) + 1
        return ex, label

    def clean_test_word(self, word):
        cleaned_word = []
        mask_idx = []
        for s in word:
            if s != " ":
                if s == "_":
                    cleaned_word.append(cfg.TOKENIZATION.mask_token)
                else:
                    cleaned_word.append(s.lower())
        for i, s in enumerate(cleaned_word):
            if s == cfg.TOKENIZATION.mask_token:
                mask_idx.append(i)
        mask_idx = torch.LongTensor(mask_idx)
        return cleaned_word, mask_idx

    def encode_one_token(self, tk):
        return self.tokens_to_ids.get(
            tk, self.tokens_to_ids[cfg.TOKENIZATION.unknown_token]
        )


def get_num_tokens_padding_idx(path):
    t = torch.load(path)
    return len(t.tokens), t.tokens_to_ids[cfg.TOKENIZATION.pad_token]

