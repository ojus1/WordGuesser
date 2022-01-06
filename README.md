# WordGuesser

## TLDR;

Input: `("N_M_STE", "A greeting in South-East Asia, particularly India.")`
Output: `NAMASTE`

## Approach

1. Use open source English dictionary to build a dataset of `(word, meaning)`.
2. During training, artificially remove letters from the ground truth.
3. Train a Masked-Language Model on the dataset.

## Results

Coming soon.
