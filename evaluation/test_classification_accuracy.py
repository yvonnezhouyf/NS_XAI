"""
Batch-test query_classification.py against labeled queries.

Modes:
  all        - run every labeled query
  sample N   - random sample of N (default 100)
  first N    - first N labeled queries (default 100)

Usage:
  python test_classification_accuracy.py all
  python test_classification_accuracy.py sample [N]
  python test_classification_accuracy.py first [N]
"""
import argparse
import csv
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))          # .../evaluation
REPO_DIR = os.path.dirname(HERE)                            # .../ns_explainer (for `core.*` imports)
sys.path.insert(0, REPO_DIR)

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)

from core.query_classification import _get_sync_client, _parse_json_response, VALID_LEVELS

import re as _re


def _parse_array_response(text: str):
    """
    Fallback parser for array format like:
      [-1, NONE]   [2, "MICRO"]   [3, MICRO]   [1, null]
    """
    m = _re.search(r"[\[\{]\s*(-?\d+)\s*,\s*([A-Za-z_\"']+|null)\s*[\]\}]", text)
    if not m:
        return None, None
    try:
        tid = int(m.group(1))
    except ValueError:
        return None, None
    raw = m.group(2).strip("\"'")
    lvl = None if raw.lower() in ("null", "none") else raw.upper()
    if lvl is not None and lvl not in VALID_LEVELS:
        lvl = None
    return tid, lvl


def _parse_any(text: str):
    tid, lvl = _parse_json_response(text)
    if tid is not None:
        return tid, lvl
    return _parse_array_response(text)
from use_cases.paratransit.config import ParatransitConfig

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)


def classify_with_retry(query: str, prompt_id: str, max_retries: int = 5):
    """
    Returns (type_id, level, error_msg).
    On transient errors (429, timeout, connection, 5xx), retries with exponential
    backoff: 1s, 2s, 4s, 8s, 16s (+ small jitter). error_msg is None on success.
    """
    client = _get_sync_client()
    delay = 1.0
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = client.responses.create(
                prompt={"id": prompt_id},
                input=[{"role": "user",
                        "content": [{"type": "input_text", "text": query}]}],
                text={"format": {"type": "text"}},
                reasoning={},
                max_output_tokens=2048,
            )
            if resp.output and resp.output[0].content:
                txt = resp.output[0].content[0].text.strip()
                tid, lvl = _parse_any(txt)
                if tid is not None:
                    return tid, lvl, None
                return None, None, f"unparseable: {txt[:120]}"
            return None, None, "empty response"
        except RETRYABLE as e:
            last_err = e
            if attempt == max_retries:
                break
            sleep_for = delay + random.uniform(0, 0.5)
            print(f"    [retry {attempt+1}/{max_retries}] {type(e).__name__}; "
                  f"sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
            delay *= 2
        except Exception as e:
            return None, None, f"{type(e).__name__}: {e}"
    return None, None, f"{type(last_err).__name__}: {last_err}"

QUERIES_PATH = os.path.join(HERE, "queries.txt")
LABELS_PATH = os.path.join(HERE, "labels.txt")
PROMPT_ID = ParatransitConfig.QUERY_CLASSIFICATION_PROMPT_ID


def load_dataset(queries_path=QUERIES_PATH, labels_path=LABELS_PATH):
    # Align by id (both files are TSV with an "id" column).
    queries_by_id = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            queries_by_id[row["id"]] = row["query"]

    dataset = []
    with open(labels_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader):
            qid = row["id"]
            if qid not in queries_by_id:
                continue
            try:
                expected = int(row["type_k"])
            except (ValueError, TypeError):
                continue
            dataset.append({
                "idx": i + 1,
                "id": qid,
                "query": queries_by_id[qid].strip(),
                "expected": expected,
            })
    return dataset


def select_items(dataset, mode, n):
    if mode == "all":
        return dataset
    if mode == "first":
        return dataset[:n]
    if mode == "sample":
        random.seed(42)
        return random.sample(dataset, min(n, len(dataset)))
    raise ValueError(f"unknown mode: {mode}")


_CACHE = {}


def load_cache(path):
    """Load prior classification results keyed by query text."""
    if not path or not os.path.exists(path):
        return {}
    cache = {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("error"):
                continue  # don't cache API failures
            if not r.get("pred"):
                continue
            try:
                pred = int(r["pred"])
            except ValueError:
                continue
            cache[r["query"].strip()] = (pred, r["level"] or None)
    return cache


def classify_one(item):
    cached = _CACHE.get(item["query"])
    if cached is not None:
        pred, level = cached
        return {**item, "pred": pred, "level": level, "error": None,
                "ok": pred == item["expected"], "cached": True}
    pred, level, err = classify_with_retry(item["query"], PROMPT_ID)
    ok = (err is None) and (pred == item["expected"])
    return {**item, "pred": pred, "level": level, "error": err,
            "ok": ok, "cached": False}


def run(items, workers=8):
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(classify_one, it): it for it in items}
        for i, fut in enumerate(as_completed(futures), 1):
            r = fut.result()
            results.append(r)
            if r.get("cached"):
                continue  # silent for cache hits
            if r["error"]:
                mark = "API"
            elif r["ok"]:
                mark = "OK "
            else:
                mark = "ERR"
            extra = f" err={r['error']}" if r["error"] else ""
            print(
                f"[{i}/{len(items)}] {mark} {r['id']} "
                f"expected={r['expected']} pred={r['pred']} "
                f"level={r['level']}{extra} | {r['query'][:80]}"
            )
    return results


def report(results, out_path):
    total = len(results)
    api_errs = sum(1 for r in results if r["error"])
    scored = [r for r in results if not r["error"]]
    correct = sum(1 for r in scored if r["ok"])
    acc = correct / len(scored) if scored else 0.0

    print("\n=== Summary ===")
    print(f"Total:       {total}")
    print(f"API errors:  {api_errs}  (excluded from accuracy)")
    print(f"Scored:      {len(scored)}")
    print(f"Correct:     {correct}")
    print(f"Accuracy:    {acc:.2%}")

    # Per-class (only over successfully-classified items)
    classes = {}
    for r in scored:
        c = classes.setdefault(r["expected"], {"n": 0, "ok": 0})
        c["n"] += 1
        c["ok"] += int(r["ok"])
    print("\nPer-label accuracy:")
    expected_labels = sorted(classes.keys())
    f1_values = []
    precision_values = []
    recall_values = []
    for k in sorted(classes.keys()):
        c = classes[k]
        print(f"  type_k={k:>3}: {c['ok']}/{c['n']} ({c['ok']/c['n']:.2%})")
        tp = sum(1 for r in scored if r["expected"] == k and r["pred"] == k)
        fp = sum(1 for r in scored if r["expected"] != k and r["pred"] == k)
        fn = sum(1 for r in scored if r["expected"] == k and r["pred"] != k)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    if expected_labels:
        print("\nMacro metrics over expected labels:")
        print(f"  Precision: {sum(precision_values)/len(precision_values):.2%}")
        print(f"  Recall:    {sum(recall_values)/len(recall_values):.2%}")
        print(f"  F1:        {sum(f1_values)/len(f1_values):.2%}")

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "idx", "expected", "pred", "level", "ok", "error", "query"])
        for r in sorted(results, key=lambda x: x["idx"]):
            w.writerow([r["id"], r["idx"], r["expected"],
                        "" if r["pred"] is None else r["pred"],
                        r["level"] or "", int(r["ok"]),
                        r["error"] or "", r["query"]])
    print(f"\nDetails written to: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["all", "sample", "first"])
    p.add_argument("n", nargs="?", type=int, default=100,
                   help="count for sample/first (default 100)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=os.path.join(HERE, "classification_results.csv"))
    p.add_argument("--queries", default=QUERIES_PATH)
    p.add_argument("--labels", default=LABELS_PATH)
    p.add_argument("--cache", default=None,
                   help="CSV of prior results; matching queries skip the API call. "
                        "Pass the path to an existing results CSV to reuse.")
    args = p.parse_args()

    if args.cache:
        _CACHE.update(load_cache(args.cache))
        print(f"Loaded {len(_CACHE)} cached classifications from {args.cache}")

    dataset = load_dataset(args.queries, args.labels)
    print(f"Loaded {len(dataset)} labeled queries.")

    items = select_items(dataset, args.mode, args.n)
    hits = sum(1 for it in items if it["query"] in _CACHE)
    print(f"Mode={args.mode} -> running {len(items)} items "
          f"({hits} cache hits, {len(items) - hits} API calls) "
          f"with {args.workers} workers. Prompt: {PROMPT_ID}")

    t0 = time.time()
    results = run(items, workers=args.workers)
    print(f"\nElapsed: {time.time() - t0:.1f}s")
    report(results, args.out)


if __name__ == "__main__":
    main()
