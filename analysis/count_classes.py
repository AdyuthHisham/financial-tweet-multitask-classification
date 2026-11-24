"""Count tickers in data/stock_tweets.csv (Kaggle 'Stock Tweets for Sentiment Analysis and Prediction')."""
import sys
import pandas as pd
from common import ROOT, dump

path = ROOT / "data" / "stock_tweets.csv"
if not path.exists():
    sys.exit(f"Download the Kaggle CSV to {path} first.")
d = pd.read_csv(path, low_memory=False)
vc = d["Stock Name"].value_counts()
dump("stock_class_counts.json", {
    "source": "data/stock_tweets.csv (Kaggle), counted by analysis/count_classes.py",
    "rows": int(len(d)), "tickers": int(vc.size),
    "counts": {k: int(v) for k, v in vc.items()},
    "share_tsla": round(float(vc["TSLA"] / len(d)), 4), "share_tsm": round(float(vc["TSM"] / len(d)), 4),
    "tickers_below_200": int((vc < 200).sum())})
