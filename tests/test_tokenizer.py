import sys

sys.path.append(".")
from core.tokenizer import CharLevelTokenizer
from core.config import cfg
import torch
import random

torch.manual_seed(123)
random.seed(123)

text = "abcde"


def test_mask_pad():
    tokenizer = CharLevelTokenizer()

    word = "testing"
    masked, masked_idx, gt = tokenizer.mask_word(word, mode="train")
    assert gt[0] == "t"
    assert gt[2] == "s"

    padded, gt_padded = tokenizer.pad(masked, gt)
    assert len(padded) == cfg.TOKENIZATION.max_seq_length - 1

    temp = "a" * 150
    temp = [tokenizer.encode_one_token(tk) for tk in temp]
    padded, _ = tokenizer.pad(temp)
    assert len(padded) == cfg.TOKENIZATION.max_seq_length - 1


def test_add_tokens():
    tokenizer = CharLevelTokenizer()
    gt_tokens = [
        cfg.TOKENIZATION.mask_token,
        cfg.TOKENIZATION.sep_token,
        cfg.TOKENIZATION.cls_token,
        cfg.TOKENIZATION.pad_token,
        cfg.TOKENIZATION.unknown_token,
        "a",
        "b",
        "c",
        "d",
        "e",
    ]
    gt_tokens_to_ids = {
        cfg.TOKENIZATION.mask_token: 0,
        cfg.TOKENIZATION.sep_token: 1,
        cfg.TOKENIZATION.cls_token: 2,
        cfg.TOKENIZATION.pad_token: 3,
        cfg.TOKENIZATION.unknown_token: 4,
        "a": 5,
        "b": 6,
        "c": 7,
        "d": 8,
        "e": 9,
    }
    gt_ids_to_tokens = {k: v for v, k in gt_tokens_to_ids.items()}

    tokenizer.add_tokens(text)
    assert set(gt_tokens) == set(tokenizer.tokens)
    for k, v in gt_tokens_to_ids.items():
        assert k in tokenizer.tokens_to_ids
        assert tokenizer.tokens_to_ids[k] == v
    for k, v in gt_ids_to_tokens.items():
        assert k in tokenizer.ids_to_tokens
        assert tokenizer.ids_to_tokens[k] == v


def test_clean_word():
    tokenizer = CharLevelTokenizer()
    tokenizer.add_tokens(text)

    test_example = "A _ b _ _ "
    cleaned, mask_idx = tokenizer.clean_test_word(test_example)
    gt = ["a", "[MASK]", "b", "[MASK]", "[MASK]"]
    assert gt == cleaned
    assert not mask_idx is None
    assert 1 in mask_idx
    assert 3 in mask_idx
    assert 4 in mask_idx


def test_encode():
    tokenizer = CharLevelTokenizer()
    tokenizer.add_tokens(text)

    word = "badef"
    description = "this is a test."
    encoded, mask_idx, label = tokenizer.encode(word, description, mode="train")

    assert not mask_idx is None
    assert len(encoded) == len(label)

    test_example = "S _ b _ _ "
    encoded, mask_idx, label = tokenizer.encode(word, description, mode="test")
    assert len(encoded) == cfg.TOKENIZATION.max_seq_length
    assert not mask_idx is None

