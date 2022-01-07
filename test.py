import gradio as gr
from core.inference import inference
from core.tokenizer import CharLevelTokenizer, get_num_tokens_padding_idx
from train import Experiment
from core.model import Net
import torch
import pandas as pd

num_tokens, padding_idx = get_num_tokens_padding_idx("./data/tokenizer.pth")
gru = Net(num_tokens, padding_idx)
tokenizer = torch.load("./data/tokenizer.pth")

model = Experiment(gru, padding_idx, num_tokens)
model.load_state_dict(torch.load("./lightning_logs/version_3/checkpoints/epoch=500-step=60035.ckpt", map_location="cpu")["state_dict"])

test = pd.read_csv("./data/NLP_Chardes_Evaluation.csv", dtype="str")

rows = []
for i in range(len(test)):
    masked_word = test.iloc[i, 0]
    description = test.iloc[i, 1]
    cleaned = inference(masked_word, description, tokenizer, model)
    rows.append((cleaned, masked_word, description))

predicted = pd.DataFrame(rows, columns=["Predicted Word", "Masked", "Meaning"])
predicted.to_csv("test_predictions.csv", index=False)