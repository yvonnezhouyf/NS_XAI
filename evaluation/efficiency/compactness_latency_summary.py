#!/usr/bin/env python3
"""
Summarize compactness and observed generation latency for evaluation outputs.

This script is intentionally read-only with respect to method outputs. It reads
the existing 240-query evidence/claims JSON files and writes summary artifacts
under evaluation/efficiency/.
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR

METHODS = {
    "ours": {
        "name": "Full NS-XAI",
        "family": "main",
        "evidence": os.path.join(EVAL_DIR, "faithfulness", "evidence_240.json"),
        "claims": os.path.join(EVAL_DIR, "faithfulness", "claims_240.json"),
    },
    "b1": {
        "name": "B1 trajectory-only",
        "family": "baseline",
        "evidence": os.path.join(EVAL_DIR, "baselines", "b1_trajectory_only", "evidence_240.json"),
        "claims": os.path.join(EVAL_DIR, "baselines", "b1_trajectory_only", "claims_240.json"),
    },
    "b2": {
        "name": "B2 trajectory + policy",
        "family": "baseline",
        "evidence": os.path.join(EVAL_DIR, "baselines", "b2_trajectory_policy", "evidence_240.json"),
        "claims": os.path.join(EVAL_DIR, "baselines", "b2_trajectory_policy", "claims_240.json"),
    },
    "b4": {
        "name": "B4/A1 single-snapshot",
        "family": "baseline_ablation",
        "evidence": os.path.join(EVAL_DIR, "baselines", "b4_single_snapshot", "evidence_240.json"),
        "claims": os.path.join(EVAL_DIR, "baselines", "b4_single_snapshot", "claims_240.json"),
    },
    "a2": {
        "name": "A2 PCTL only",
        "family": "ablation",
        "evidence": os.path.join(EVAL_DIR, "ablations", "a2_pctl_only", "evidence_240.json"),
        "claims": os.path.join(EVAL_DIR, "ablations", "a2_pctl_only", "claims_240.json"),
    },
    "a3": {
        "name": "A3 derived only",
        "family": "ablation",
        "evidence": os.path.join(EVAL_DIR, "ablations", "a3_derived_only", "evidence_240.json"),
        "claims": os.path.join(EVAL_DIR, "ablations", "a3_derived_only", "claims_240.json"),
    },
    "a4": {
        "name": "A4 fixed reference",
        "family": "ablation",
        "evidence": os.path.join(EVAL_DIR, "ablations", "a4_fixed_reference", "evidence_240.json"),
        "claims": os.path.join(EVAL_DIR, "ablations", "a4_fixed_reference", "claims_240.json"),
    },
}

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[.'_-][A-Za-z0-9]+)?")


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def word_count(text: Optional[str]) -> int:
    return len(WORD_RE.findall(text or ""))


def char_count(text: Optional[str]) -> int:
    return len(text or "")


def to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def summary(values: Iterable[Optional[float]]) -> Dict[str, Any]:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
            "min": None,
            "max": None,
            "q25": None,
            "q75": None,
        }
    vals_sorted = sorted(vals)
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "q25": vals_sorted[int(0.25 * (len(vals_sorted) - 1))],
        "q75": vals_sorted[int(0.75 * (len(vals_sorted) - 1))],
    }


def claim_metrics(claim_record: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    claims = (claim_record or {}).get("claims") or []
    kind_counts = Counter(c.get("kind") or "unknown" for c in claims if isinstance(c, dict))
    numeric_values = []
    for claim in claims:
        if not isinstance(claim, dict) or claim.get("kind") != "numerical":
            continue
        value = to_float(claim.get("value"))
        if value is not None:
            numeric_values.append(round(value, 4))
    return {
        "claim_count": len(claims),
        "numerical_claim_count": kind_counts.get("numerical", 0),
        "categorical_claim_count": kind_counts.get("categorical", 0),
        "comparative_claim_count": kind_counts.get("comparative", 0),
        "causal_claim_count": kind_counts.get("causal", 0),
        "unique_numeric_values": len(set(numeric_values)),
    }


def load_method_records(method_id: str, cfg: Dict[str, str]) -> List[Dict[str, Any]]:
    evidence_records = read_json(cfg["evidence"])
    claim_records = read_json(cfg["claims"])
    claims_by_id = {
        r.get("query_id"): r for r in claim_records
        if isinstance(r, dict) and r.get("query_id")
    }
    rows = []
    for rec in evidence_records:
        if not isinstance(rec, dict) or not rec.get("query_id"):
            continue
        qid = rec["query_id"]
        claim_row = claims_by_id.get(qid)
        explanation = rec.get("explanation") or ""
        llm_input = rec.get("llm_input") or ""
        cm = claim_metrics(claim_row)
        rows.append({
            "method": method_id,
            "method_name": cfg["name"],
            "family": cfg["family"],
            "query_id": qid,
            "type_k": rec.get("type_k"),
            "scenario": rec.get("scenario"),
            "status": rec.get("status"),
            "claim_status": (claim_row or {}).get("status"),
            "latency_sec": to_float(rec.get("elapsed_sec")),
            "explanation_words": word_count(explanation),
            "explanation_chars": char_count(explanation),
            "input_words": word_count(llm_input),
            "input_chars": char_count(llm_input),
            **cm,
        })
    return rows


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "records": len(rows),
        "ok_records": sum(1 for r in rows if r.get("status") == "ok"),
        "claim_ok_records": sum(1 for r in rows if r.get("claim_status") == "ok"),
        "latency_sec": summary(r.get("latency_sec") for r in rows),
        "explanation_words": summary(r.get("explanation_words") for r in rows),
        "explanation_chars": summary(r.get("explanation_chars") for r in rows),
        "input_words": summary(r.get("input_words") for r in rows),
        "claim_count": summary(r.get("claim_count") for r in rows),
        "numerical_claim_count": summary(r.get("numerical_claim_count") for r in rows),
        "unique_numeric_values": summary(r.get("unique_numeric_values") for r in rows),
    }


def grouped_summary(rows: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    def sort_key(item):
        k = item[0]
        try:
            return (0, int(k))
        except ValueError:
            return (1, k)
    return {k: summarize_rows(v) for k, v in sorted(groups.items(), key=sort_key)}


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def flatten_method_summary(method_id: str, cfg: Dict[str, str],
                           s: Dict[str, Any]) -> Dict[str, Any]:
    def m(metric: str, stat: str) -> Any:
        return s[metric].get(stat)
    return {
        "method": method_id,
        "method_name": cfg["name"],
        "family": cfg["family"],
        "records": s["records"],
        "ok_records": s["ok_records"],
        "claim_ok_records": s["claim_ok_records"],
        "latency_mean_sec": m("latency_sec", "mean"),
        "latency_median_sec": m("latency_sec", "median"),
        "latency_q25_sec": m("latency_sec", "q25"),
        "latency_q75_sec": m("latency_sec", "q75"),
        "explanation_words_mean": m("explanation_words", "mean"),
        "explanation_words_median": m("explanation_words", "median"),
        "input_words_mean": m("input_words", "mean"),
        "claim_count_mean": m("claim_count", "mean"),
        "claim_count_median": m("claim_count", "median"),
        "numerical_claim_count_mean": m("numerical_claim_count", "mean"),
        "unique_numeric_values_mean": m("unique_numeric_values", "mean"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                   choices=list(METHODS.keys()))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    method_summaries: Dict[str, Any] = {}
    for method_id in args.methods:
        cfg = METHODS[method_id]
        rows = load_method_records(method_id, cfg)
        all_rows.extend(rows)
        method_summaries[method_id] = {
            "name": cfg["name"],
            "family": cfg["family"],
            "overall": summarize_rows(rows),
            "by_type_k": grouped_summary(rows, "type_k"),
            "by_scenario": grouped_summary(rows, "scenario"),
            "inputs": {
                "evidence": cfg["evidence"],
                "claims": cfg["claims"],
            },
        }

    artifact = {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "notes": {
            "latency_sec": (
                "Record-level elapsed_sec from each generator. This is observed "
                "wall-clock time for that generator's record path, not a clean "
                "hardware-normalized stage profile."
            ),
            "compactness": (
                "Word counts are regex-tokenized English words/numbers in the "
                "generated explanation. Claim counts come from existing "
                "claim_extractor outputs."
            ),
        },
        "methods": method_summaries,
    }

    summary_rows = [
        flatten_method_summary(method_id, METHODS[method_id],
                               method_summaries[method_id]["overall"])
        for method_id in args.methods
    ]

    per_record_fields = [
        "method", "method_name", "family", "query_id", "type_k", "scenario",
        "status", "claim_status", "latency_sec", "explanation_words",
        "explanation_chars", "input_words", "input_chars", "claim_count",
        "numerical_claim_count", "categorical_claim_count",
        "comparative_claim_count", "causal_claim_count",
        "unique_numeric_values",
    ]
    summary_fields = list(summary_rows[0].keys()) if summary_rows else []

    json_path = os.path.join(args.output_dir, "compactness_latency_summary.json")
    per_method_csv = os.path.join(args.output_dir, "compactness_latency_by_method.csv")
    per_record_csv = os.path.join(args.output_dir, "compactness_latency_per_record.csv")

    write_json(json_path, artifact)
    write_csv(per_method_csv, summary_rows, summary_fields)
    write_csv(per_record_csv, all_rows, per_record_fields)

    print("=== Compactness + Latency Summary ===")
    for row in summary_rows:
        print(
            f"{row['method']:>4} | words={row['explanation_words_mean']:.1f} "
            f"claims={row['claim_count_mean']:.1f} "
            f"nums={row['unique_numeric_values_mean']:.1f} "
            f"latency={row['latency_mean_sec']:.2f}s"
        )
    print(f"JSON       -> {json_path}")
    print(f"By method  -> {per_method_csv}")
    print(f"Per record -> {per_record_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
