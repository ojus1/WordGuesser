import gradio as gr
from core.inference import inference
from core.tokenizer import CharLevelTokenizer, get_num_tokens_padding_idx
from train import Experiment
from core.model import Net
import torch
from flask import Flask, jsonify, request
import json

num_tokens, padding_idx = get_num_tokens_padding_idx("./data/tokenizer.pth")
gru = Net(num_tokens, padding_idx)
tokenizer = torch.load("./data/tokenizer.pth")

model = Experiment(gru, padding_idx, num_tokens)
model.load_state_dict(torch.load("./lightning_logs/version_3/checkpoints/epoch=500-step=60035.ckpt", map_location="cpu")["state_dict"])

app = Flask(__name__)

@app.route('/guess_word', methods = ['POST'])
def spell_correct():
    j = json.loads(request.data)
    cleaned = inference(j["masked_word"], j["description"], tokenizer, model)
    return jsonify({'predicted_word': cleaned})


if __name__ == '__main__':
	app.run(debug = True)
'''
curl --header "Content-Type: application/json" \
--request POST \
--data '{"masked_word":"DEM_G_A_HY","description":"the statistical study of populations."}' \
http://127.0.0.1:5000/guess_word
'''