"""Parse the saved cell outputs of notebooks/ApproachB.ipynb into JSON. Nothing is re-trained.

Task naming in the notebook: TaskA = company ("Class"), TaskB = sentiment. Confirmed in
test_model_single, where acc_taskA uses label_clas and acc_taskB uses label_sen.
Report tables are copied by hand from docs/report numbers (Tables I and II) and marked as such.
"""
import json
import re
from common import ROOT, dump

nb = json.loads((ROOT / "notebooks" / "ApproachB.ipynb").read_text())
outs = {}
for i, c in enumerate(nb["cells"]):
    for o in c.get("outputs", []):
        if "text" in o:
            outs.setdefault(i, "")
            outs[i] += "".join(o["text"])
text_of = lambda marker: next(t for t in outs.values() if marker in t)

# Phase 1: head-only training, 30 epochs, val accuracy of company head for copies A and B.
head = text_of("Epoch 1/30")
h = re.findall(r"Epoch (\d+)/30.*?Val Loss - A: ([\d.]+), B: ([\d.]+)\s+Val Accuracy - A TaskA: ([\d.]+), B TaskA: ([\d.]+)", head, re.S)
dump("approachB_head_phase.json", {
    "source": "notebooks/ApproachB.ipynb, stdout of the head-training cell",
    "epochs": [int(x[0]) for x in h],
    "val_loss_L": [float(x[1]) for x in h], "val_loss_R": [float(x[2]) for x in h],
    "val_acc_company_L": [float(x[3]) for x in h], "val_acc_company_R": [float(x[4]) for x in h]})

# Phase 2: full fine-tuning, 20 epochs; L and R are the two network copies.
full = text_of("Epoch 1/20")
f = re.findall(r"Epoch (\d+)/20.*?Train Loss: ([\d.]+) \| Val Loss: ([\d.]+)\s+L\[TaskA\]: ([\d.]+), L\[TaskB\]: ([\d.]+), R\[TaskA\]: ([\d.]+), R\[TaskB\]: ([\d.]+)", full, re.S)
dump("approachB_full_phase.json", {
    "source": "notebooks/ApproachB.ipynb, stdout of the full-training cell (validation accuracy)",
    "epochs": [int(x[0]) for x in f],
    "train_loss": [float(x[1]) for x in f], "val_loss": [float(x[2]) for x in f],
    "val_acc_company_L": [float(x[3]) for x in f], "val_acc_sentiment_L": [float(x[4]) for x in f],
    "val_acc_company_R": [float(x[5]) for x in f], "val_acc_sentiment_R": [float(x[6]) for x in f]})

# Test-set evaluation of every saved checkpoint.
ev = text_of("Evaluating best_model_A.pt")
test = {}
for m in re.finditer(r"Model (\S+?):\s+Accuracy → Class=([\d.]+), Sent=([\d.]+), Avg=([\d.]+)\s+F1 \(macro\)\s+→ Class=([\d.]+), Sent=([\d.]+), Avg=([\d.]+)\s+F1 \(weighted\)→ Class=([\d.]+), Sent=([\d.]+)", ev):
    g = m.groups()
    test[g[0]] = {"accuracy": {"company": float(g[1]), "sentiment": float(g[2])},
                  "f1_macro": {"company": float(g[4]), "sentiment": float(g[5])},
                  "f1_weighted": {"company": float(g[7]), "sentiment": float(g[8])}}
dump("approachB_test.json", {"source": "notebooks/ApproachB.ipynb, test-set evaluation cell", "checkpoints": test,
                             "selected": "final_best_model_A.pt (best by accuracy and by macro F1 in the notebook)"})

# Report tables, typed from Project_report.pdf Tables I and II (not recomputed here).
dump("report_tables.json", {
    "source": "Project report, Tables I and II (typed by hand; Approach A has no saved outputs)",
    "sentiment": {"FinBERT (A)": {"accuracy": 0.80, "f1_macro": 0.80, "f1_weighted": 0.80},
                  "Multi-task (B)": {"accuracy": 0.83, "f1_macro": 0.83, "f1_weighted": 0.83}},
    "company": {"BERTweet (A)": {"accuracy": 0.60, "f1_macro": 0.67, "f1_weighted": 0.60},
                "Multi-task (B)": {"accuracy": 0.68, "f1_macro": 0.68, "f1_weighted": 0.65}}})
print(len(h), len(f), len(test))
assert len(h) == 30 and len(f) == 20 and len(test) >= 6
