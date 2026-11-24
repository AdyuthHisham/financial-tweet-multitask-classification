# Method notes

## Data
- Sentiment: `StephanAkkerman/financial-tweets` (Hugging Face). Labels Bearish, Neutral, Bullish, mapped to 0, 1, 2.
- Company: Kaggle "Stock Tweets for Sentiment Analysis and Prediction". Label is the ticker and company name together.
- Each dataset is balanced by sampling an equal number of rows per class without replacement. Companies with fewer than 200 rows become `OTHERS`.
- Tweets are normalised with `src/TweetNormalizer.py`.

## Approach A
FinBERT (sentiment) and BERTweet (company) are fine-tuned separately: 80/20 stratified split, 96 tokens, AdamW, cosine schedule, early stopping on F1 macro.
A helper function runs both on one tweet.

## Approach B
- Encoder: FinTwitBERT. Heads: the original sentiment head, and a new two-layer company head (768 to 1024 to 21, ReLU, dropout 0.2) trained from scratch.
- Datasets are concatenated. A row carries one label and the other is set to an ignore value (-10).
- Two copies (L, R) are trained with a loss `L_GT + lambda * L_semi`, lambda = 1. `L_GT` is cross entropy where a label exists. `L_semi` is a KL consistency term between the copies where it does not. This follows Knowledge Assembly (Spinola et al., arXiv 2306.08839).
- Phase 1: encoder and sentiment head frozen, company head only, lr 5e-3, 30 epochs, batch 64, class-weighted cross entropy.
- Phase 2: everything trained, heads lr 1e-4, encoder lr 1e-5, 20 epochs, batch 32.
- Split: 75% train, 12.5% validation, 12.5% test, made separately per dataset and then concatenated.
- The better of the two copies on validation is kept for inference.
