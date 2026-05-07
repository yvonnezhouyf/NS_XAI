"""
Compute accuracy, macro-F1, per-class precision/recall/F1, and confusion matrix
from classification_results.csv.

Usage:
  python compute_metrics.py [results.csv]
"""
import csv
import os
import sys
from collections import defaultdict


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue
            try:
                rows.append((int(r["expected"]), int(r["pred"])))
            except (ValueError, TypeError):
                continue
    return rows


def per_class_prf(rows):
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    support = defaultdict(int)
    for y_true, y_pred in rows:
        support[y_true] += 1
        if y_true == y_pred:
            tp[y_true] += 1
        else:
            fn[y_true] += 1
            fp[y_pred] += 1

    classes = sorted(set(support) | set(fp))
    result = {}
    for c in classes:
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        result[c] = {"precision": p, "recall": r, "f1": f1, "support": support[c]}
    return result


def confusion_matrix(rows, classes):
    cm = {c: {c2: 0 for c2 in classes} for c in classes}
    for y_true, y_pred in rows:
        if y_true in cm and y_pred in cm[y_true]:
            cm[y_true][y_pred] += 1
    return cm


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "classification_results.csv"
    if not os.path.exists(path):
        print(f"file not found: {path}")
        sys.exit(1)

    rows = load(path)
    total = len(rows)
    correct = sum(1 for y, p in rows if y == p)
    acc = correct / total if total else 0.0

    prf = per_class_prf(rows)
    classes = sorted(prf.keys())

    # Macro / Weighted F1
    macro_f1 = sum(prf[c]["f1"] for c in classes) / len(classes)
    total_support = sum(prf[c]["support"] for c in classes)
    weighted_f1 = sum(prf[c]["f1"] * prf[c]["support"] for c in classes) / total_support

    macro_p = sum(prf[c]["precision"] for c in classes) / len(classes)
    macro_r = sum(prf[c]["recall"] for c in classes) / len(classes)

    print(f"=== Overall ===")
    print(f"Total scored: {total}")
    print(f"Accuracy:     {acc:.4f}  ({correct}/{total})")
    print(f"Macro-P:      {macro_p:.4f}")
    print(f"Macro-R:      {macro_r:.4f}")
    print(f"Macro-F1:     {macro_f1:.4f}")
    print(f"Weighted-F1:  {weighted_f1:.4f}")

    print(f"\n=== Per-class ===")
    print(f"{'class':>6}  {'prec':>7} {'recall':>7} {'f1':>7} {'support':>7}")
    for c in classes:
        s = prf[c]
        print(f"{c:>6}  {s['precision']:>7.4f} {s['recall']:>7.4f} "
              f"{s['f1']:>7.4f} {s['support']:>7}")

    print(f"\n=== Confusion matrix (rows=true, cols=pred) ===")
    cm = confusion_matrix(rows, classes)
    header = "       " + " ".join(f"{c:>5}" for c in classes)
    print(header)
    for true_c in classes:
        row = " ".join(f"{cm[true_c][pred_c]:>5}" for pred_c in classes)
        total_row = sum(cm[true_c].values())
        print(f"  {true_c:>3}: {row}   (n={total_row})")


if __name__ == "__main__":
    main()
