PY ?= python
# Figures read the notebook outputs and docs/metrics/*.json; no GPU or dataset needed.
figures:
	$(PY) analysis/extract_metrics.py
	$(PY) analysis/fig_class_balance.py
	$(PY) analysis/fig_training.py
	$(PY) analysis/fig_results.py
	$(PY) analysis/fig_architecture.py

# Optional: recount classes from data/stock_tweets.csv (Kaggle download, not shipped).
counts:
	$(PY) analysis/count_classes.py
