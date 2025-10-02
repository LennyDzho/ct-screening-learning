import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, average_precision_score

from ctlib.config.paths import BASE_DIR

# путь к предсказаниям (замени на свой)
csv_path = BASE_DIR / "runs_binary" / "binary_r3d18" / "epochs" / "epoch_019" / "val_preds.csv"

df = pd.read_csv(csv_path)
y_true = df["label"].values
y_prob = df["prob"].values

# ===== ROC-кривая =====
fpr, tpr, thresholds_roc = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC AUC = {roc_auc:.3f}")
plt.plot([0,1],[0,1], color="gray", lw=1, linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid()
plt.show()

# ===== PR-кривая =====
prec, rec, thresholds_pr = precision_recall_curve(y_true, y_prob)

plt.figure(figsize=(6,6))
plt.plot(rec, prec, color="green", lw=2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid()
plt.show()

# ===== Поиск оптимального порога =====
best_f1, best_thr = 0, 0.5
for thr in thresholds_pr:
    preds = (y_prob >= thr).astype(int)
    f1 = f1_score(y_true, preds)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr


y_true = df["label"].values
y_prob = df["prob"].values

prec, rec, thr = precision_recall_curve(y_true, y_prob)
ap = average_precision_score(y_true, y_prob)

plt.figure(figsize=(6,6))
plt.plot(rec, prec, color="green", lw=2, label=f"AP = {ap:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid()
plt.show()

print(f"Лучший порог по F1 = {best_thr:.3f}, F1 = {best_f1:.3f}")
