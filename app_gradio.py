import gradio as gr
from core.inference import inference
from core.tokenizer import CharLevelTokenizer, get_num_tokens_padding_idx
from train import Experiment
from core.model import Net
import torch

num_tokens, padding_idx = get_num_tokens_padding_idx("./data/tokenizer.pth")
gru = Net(num_tokens, padding_idx)
tokenizer = torch.load("./data/tokenizer.pth")

model = Experiment(gru, padding_idx, num_tokens)
model.load_state_dict(torch.load("./lightning_logs/version_3/checkpoints/epoch=500-step=60035.ckpt", map_location="cpu")["state_dict"])

def spell_correct(masked_word, description):
    cleaned = inference(masked_word, description, tokenizer, model)
    return cleaned

app = gr.Interface(
    fn=spell_correct, 
    inputs=["text", "text"],
    outputs="text"
)
app.launch()