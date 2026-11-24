# notebooks

`ApproachA.ipynb` (two separate fine-tuned models) and `ApproachB.ipynb` (one shared encoder, two heads).
Both are the original course-project notebooks, kept with their saved outputs. Approach A has no saved outputs.

To re-run them you need a GPU, the Kaggle CSV at `dataset/stock_tweets.csv`, `src/TweetNormalizer.py` and `src/class_mapping.json`
next to the notebook (Approach A reads the CSV from the notebook folder itself), and network access for the Hugging Face datasets and checkpoints. A local path in one output cell was replaced with `<local-path>`.
