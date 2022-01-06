import pandas as pd
import numpy as np
from core.tokenizer import CharLevelTokenizer
from argparse import ArgumentParser
import torch

parser = ArgumentParser()
parser.add_argument(
    "--dataset_csv_path",
    default="./data/train.csv",
    help="The csv must have two columns, words and their description. Default: uses open-source english dictionary.",
)
args = parser.parse_args()

train = pd.read_csv(args.dataset_csv_path)
train = train.dropna(axis=0)
# Find unique characters and add those to the tokenizer
unique_chars = set()
for i in range(len(train)):
    for j in range(2):
        for c in train.iloc[i, j]:
            unique_chars.add(c)

tokenizer = CharLevelTokenizer()
tokenizer.add_tokens(list(unique_chars))
torch.save(tokenizer, "./data/tokenizer.pth")
print("Tokenizer dumped to ./data/tokenizer.pth")
