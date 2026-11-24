# Results, data sources and mismatches

## Data sources

| Figure / number | File | Source |
|---|---|---|
| `media/class_balance.png` | `docs/metrics/stock_class_counts.json` | Counted from the Kaggle CSV by `analysis/count_classes.py` (`make counts`) |
| `media/training_curves.png`, `media/hero.gif` | `approachB_head_phase.json`, `approachB_full_phase.json` | Parsed from saved cell outputs of `notebooks/ApproachB.ipynb` by `analysis/extract_metrics.py` |
| Approach B test scores | `approachB_test.json` | Same notebook, test-set evaluation cell |
| `media/results.png` (Approach A bars, and the B bars) | `report_tables.json` | Typed from Tables I and II of the project report. `ApproachA.ipynb` has no saved outputs, so Approach A cannot be checked from the repo |
| `media/architecture.svg` | none | Drawn by `analysis/fig_architecture.py` from the method description |

Nothing was re-trained for this repo. The notebooks need a GPU and downloads.

## Test scores (Approach B, selected checkpoint `final_best_model_A.pt`)

| Task | Accuracy | F1 macro | F1 weighted |
|---|---|---|---|
| Sentiment | 0.8351 | 0.8317 | 0.8315 |
| Company | 0.6804 | 0.6841 | 0.6485 |

These match Tables I and II of the report for the multi-task model after rounding.
The notebook also evaluated five other checkpoints (see `approachB_test.json`).
`best_model_A.pt` and `best_model_B.pt` come from the head-only phase, where the sentiment head is not trained further, and score about 0.27 on sentiment.
They are not the final models.

## Mismatches found

1. **BERTweet company score.** Report Table II gives BERTweet accuracy 0.60 and F1 macro 0.67. The abstract, the poster and the old portfolio text quote "67%" for BERTweet next to 68% for the multi-task model.
   The 67% is F1 macro. The multi-task model's accuracy is 0.68, so the like-for-like accuracy gain is 0.60 to 0.68.
2. **Size of the gain.** The results text says the multi-task model is "0.03% / 0.08% better". The tables show 3 and 8 percentage points (0.80 to 0.83, 0.60 to 0.68). Both are single runs, so we do not claim the gain is statistically meaningful.
3. **Kaggle row count.** The report and poster say 64,479 tweets. The CSV used in the notebooks has 80,793 rows (25 tickers). Ticker shares match the report (TSLA 46.3%, TSM 13.7%).
4. **Class count.** 25 tickers in the raw data, 5 of them under 200 rows, which gives the 20 named companies plus `OTHERS` in `src/class_mapping.json`.
5. **Task names in the notebook.** `TaskA` is company and `TaskB` is sentiment (checked in `test_model_single`). The head-phase log has only `TaskA` because only the company head is trained.
6. **Sentiment dataset size.** The report says the sentiment set was pruned to 44,426 rows. That dataset is downloaded from Hugging Face at run time, so this number is not recomputed here.
