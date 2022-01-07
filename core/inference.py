from .config import cfg
import torch


@torch.no_grad()
def inference(masked_word, description, tokenizer, model):
    example, mask_idx, _ = tokenizer.encode(masked_word, description, mode="test")
    example = torch.LongTensor(example).unsqueeze(0)
    preds = model(example).argmax(-1).squeeze(0)

    cleaned_word = masked_word.replace(" ", "")
    cleaned_word = [c for c in cleaned_word]
    for i in mask_idx:
        decoded_char = tokenizer.ids_to_tokens[preds[i-1].item()]
        cleaned_word[i - 1] = decoded_char
    return "".join(cleaned_word).lower()

