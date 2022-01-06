import json
import pandas as pd
from core.config import cfg
import numpy as np

dict_path = "data/eng_dict.json"  # source: https://github.com/adambom/dictionary/blob/master/dictionary.json

raw = json.load(open(dict_path, "r"))

word_lengths = []
meaning_lengths = []

rows = []
for word, meanings in raw.items():
    # Pick only one-word phrases only
    word = word.lower()
    word_lengths.append(len(word))
    meanings = meanings.split(";")
    if word.isalpha() and len(word.split(" ")) == 1:
        for meaning in meanings:
            meaning_lengths.append(len(meaning))
            # Make new row for each meaning of the word
            rows.append([word, meaning])

print("Character Lengths")
print(f"Words: Mean={np.mean(word_lengths)}, Std={np.std(word_lengths)}")
print(f"Meaning: Mean={np.mean(meaning_lengths)}, Std={np.std(meaning_lengths)}")

train = pd.DataFrame(rows, columns=["Masked", "Meaning"])

train.to_csv("./data/train.csv", index=False)
print("Processed training data.")
