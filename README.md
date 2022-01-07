# WordGuesser

## TLDR;

Input: `("N_M_STE", "A greeting in South-East Asia, particularly India.")`

Output: `NAMASTE`

## Approach

1. Use open source English dictionary to build a dataset of `(word, meaning)`.
2. During training, artificially remove letters from the ground truth.
3. Train a Masked-Language Model on the dataset.

## Usage

### Training a model from Scratch

1. Run `python3 train.py --dataset_csv_path path/to/dataset.csv`, the csv must contain words in the first column, and their meanings in the second column.

For example, run `python3 train.py --dataset_csv_path ./data/train.csv` to train on English words (dataset included).

OR 

1. Build docker image with `python3 build_container.py train`
2. Run `sudo docker run train path/to/dataset.csv`

### Gradio app

1. Run `python3 app_gradio.py` to launch gradio-based web-app.

OR 

1. Build docker image with `python3 build_container.py inference_gradio`
2. Run `sudo docker run inference_gradio`

### Rest API

1. Run `python3 app_rest.py` to launch flask REST server.

OR 

1. Build docker image with `python3 build_container.py inference_rest`
2. Run `sudo docker run inference_rest`

Example request

```
curl --header "Content-Type: application/json" \
--request POST \
--data '{"masked_word":"DEM_G_A_HY","description":"the statistical study of populations."}' \
http://127.0.0.1:5000/guess_word
```

## Project Structure

1. `core` contains implementations of the dataset, model, tokenizer etc.
2. `data` contains csvs for training, testing, dumped tokenizer etc.
3. `tests` contains unit tests for modules in `core`
4. `build_tokenizer.py` builds a character-level tokenizer for a given CSV.
5. `clean_data.py` is used for cleaning the open-source dataset used.
6. `app.py` is a simple gradio inference for inference with a trained model (included).
7. `app.py` is a simple REST API (flask).
8. `train.py` is the training script, uses PyTorch-Lightning, pass `--dataset_csv_path` argument to the script (works for the docker container as well).
9. `docker_files` contain `Dockerfile`s for inference and training.
10. `build_container.py` builds docker containers for inference_gradio, inference_rest, and inference_training. Example: `python3 build_container.py inference_gradio`.