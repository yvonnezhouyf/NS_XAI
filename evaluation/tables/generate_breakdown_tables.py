#!/usr/bin/env python3
"""
Generate appendix-ready per-type and per-scenario breakdown tables.

Inputs are existing graded JSON files. The script does not call any LLMs and
does not modify evaluation outputs.
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)

METHODS = {
    "ours": {
        "name": "Full NS-XAI",
        "family": "main",
        "faithfulness": os.path.join(EVAL_DIR, "faithfulness", "graded_claims_240.json"),
        "delta": os.path.join(EVAL_DIR, "delta_recovery", "ours_graded_240.json"),
    },
    "b1": {
        "name": "B1 trajectory-only",
        "family": "baseline",
        "faithfulness": os.path.join(EVAL_DIR, "baselines", "b1_trajectory_only", "graded_claims_240.json"),
        "delta": os.path.join(EVAL_DIR, "delta_recovery", "b1_graded_240.json"),
    },
    "b2": {
        "name": "B2 trajectory + policy",
        "family": "baseline",
        "faithfulness": os.path.join(EVAL_DIR, "baselines", "b2_trajectory_policy", "graded_claims_240.json"),
        "delta": os.path.join(EVAL_DIR, "delta_recovery", "b2_graded_240.json"),
    },
    "b4": {
        "name": "B4/A1 single-snapshot",
        "family": "baseline_ablation",
        "faithfulness": os.path.join(EVAL_DIR, "baselines", "b4_single_snapshot", "graded_claims_240.json"),
        "delta": os.path.join(EVAL_DIR, "delta_recovery", "b4_graded_240.json"),
    },
    "a2": {
        "name": "A2 PCTL only",
        "family": "ablation",
        "faithfulness": os.path.join(EVAL_DIR, "ablations", "a2_pctl_only", "graded_claims_240.json"),
        "delta": os.path.join(EVAL_DIR, "delta_recovery", "a2_graded_240.json"),
    },
    "a3": {
        "name": "A3 derived only",
        "family": "ablation",
        "faithfulness": os.path.join(EVAL_DIR, "ablations", "a3_derived_only", "graded_claims_240.json"),
        "delta": os.path.join(EVAL_DIR, "delta_recovery", "a3_graded_240.json"),
    },
    "a4": {
        "name": "A4 fixed reference",
        "family": "ablation",
        "faithfulness": os.path.join(EVAL_DIR, "ablations", "a4_fixed_reference", "graded_claims_240.json"),
        "delta": os.path.join(EVAL_DIR, "delta_recovery", "a4_graded_240.json"),
    },
}

VERDICTS = ("V+", "V-", "U", "N/A")


def read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def sort_value(value: Any) -> Tuple[int, Any]:
    if value in (None, "", "all"):
        return (0, "")
    try:
        return (1, int(value))
    except (TypeError, ValueError):
        return (2, str(value))


def rate(num: int, den: int) -> Optional[float]:
    return num / den if den else None


def new_faith_acc(method_id: str, cfg: Dict[str, str], group: str,
                  group_value: Any, kind: str) -> Dict[str, Any]:
    return {
        "method": method_id,
        "method_name": cfg["name"],
        "family": cfg["family"],
        "group": group,
        "group_value": group_value,
        "kind": kind,
        "query_ids": set(),
        "V+": 0,
        "V-": 0,
        "U": 0,
        "N/A": 0,
    }


def add_claim(acc: Dict[str, Any], qid: str, verdict: str) -> None:
    acc["query_ids"].add(qid)
    if verdict not in VERDICTS:
        verdict = "U"
    acc[verdict] += 1


def finalize_faith_row(row: Dict[str, Any]) -> Dict[str, Any]:
    graded_total = row["V+"] + row["V-"] + row["U"]
    claim_total = graded_total + row["N/A"]
    out = {k: v for k, v in row.items() if k != "query_ids"}
    out["records"] = len(row["query_ids"])
    out["claim_total"] = claim_total
    out["graded_total"] = graded_total
    out["v_pos_rate"] = rate(row["V+"], graded_total)
    out["v_neg_rate"] = rate(row["V-"], graded_total)
    out["u_rate"] = rate(row["U"], graded_total)
    out["na_rate_all_claims"] = rate(row["N/A"], claim_total)
    return out


def collect_faithfulness_rows(method_id: str, cfg: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    data = read_json(cfg["faithfulness"])
    records = data.get("records", [])
    tables = {
        "overall": {},
        "by_kind": {},
        "by_type": {},
        "by_scenario": {},
        "by_type_kind": {},
        "by_scenario_kind": {},
    }

    def get_acc(table: str, group: str, group_value: Any, kind: str) -> Dict[str, Any]:
        key = (str(group_value), kind)
        if key not in tables[table]:
            tables[table][key] = new_faith_acc(method_id, cfg, group, group_value, kind)
        return tables[table][key]

    for rec in records:
        qid = rec.get("query_id")
        type_k = rec.get("type_k")
        scenario = rec.get("scenario")
        for item in rec.get("graded_claims", []) or []:
            claim = item.get("claim") or {}
            kind = claim.get("kind") or "unknown"
            verdict = item.get("verdict") or "U"

            add_claim(get_acc("overall", "overall", "all", "all"), qid, verdict)
            add_claim(get_acc("by_kind", "kind", kind, kind), qid, verdict)
            add_claim(get_acc("by_type", "type_k", type_k, "all"), qid, verdict)
            add_claim(get_acc("by_scenario", "scenario", scenario, "all"), qid, verdict)
            add_claim(get_acc("by_type_kind", "type_k", type_k, kind), qid, verdict)
            add_claim(get_acc("by_scenario_kind", "scenario", scenario, kind), qid, verdict)

    finalized = {}
    for name, table in tables.items():
        rows = [finalize_faith_row(v) for v in table.values()]
        rows.sort(key=lambda r: (r["method"], sort_value(r["group_value"]), r["kind"]))
        finalized[name] = rows
    return finalized


DELTA_FIELDS = [
    "records",
    "records_with_targets",
    "delta_targets",
    "delta_recovered",
    "delta_coverage_rate",
    "sign_matches",
    "sign_match_rate_all",
    "sign_match_rate_recovered",
    "magnitude_n",
    "magnitude_mae",
    "spearman_n",
    "spearman",
    "scalar_targets",
    "scalar_recovered",
    "scalar_coverage_rate",
    "scalar_within_tolerance",
    "scalar_within_tolerance_rate",
    "scalar_mae",
]


def flatten_delta_summary(method_id: str, cfg: Dict[str, str], group: str,
                          group_value: Any, stats: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "method": method_id,
        "method_name": cfg["name"],
        "family": cfg["family"],
        "group": group,
        "group_value": group_value,
        "status_counts": json.dumps(stats.get("status_counts", {}), sort_keys=True),
    }
    for field in DELTA_FIELDS:
        row[field] = stats.get(field)
    return row


def collect_delta_rows(method_id: str, cfg: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    data = read_json(cfg["delta"])
    summary = data.get("summary", {})
    rows = {
        "overall": [
            flatten_delta_summary(method_id, cfg, "overall", "all", summary.get("overall", {}))
        ],
        "by_type": [],
        "by_scenario": [],
    }
    for type_k, stats in (summary.get("by_type_k") or {}).items():
        rows["by_type"].append(flatten_delta_summary(method_id, cfg, "type_k", type_k, stats))
    for scenario, stats in (summary.get("by_scenario") or {}).items():
        rows["by_scenario"].append(flatten_delta_summary(method_id, cfg, "scenario", scenario, stats))
    for key in ("by_type", "by_scenario"):
        rows[key].sort(key=lambda r: (r["method"], sort_value(r["group_value"])))
    return rows


def extend_tables(target: Dict[str, List[Dict[str, Any]]],
                  source: Dict[str, List[Dict[str, Any]]]) -> None:
    for name, rows in source.items():
        target[name].extend(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default=SCRIPT_DIR)
    p.add_argument("--methods", nargs="+", default=list(METHODS.keys()),
                   choices=list(METHODS.keys()))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    faith_tables = {
        "overall": [],
        "by_kind": [],
        "by_type": [],
        "by_scenario": [],
        "by_type_kind": [],
        "by_scenario_kind": [],
    }
    delta_tables = {
        "overall": [],
        "by_type": [],
        "by_scenario": [],
    }

    for method_id in args.methods:
        cfg = METHODS[method_id]
        extend_tables(faith_tables, collect_faithfulness_rows(method_id, cfg))
        extend_tables(delta_tables, collect_delta_rows(method_id, cfg))

    faith_fields = [
        "method", "method_name", "family", "group", "group_value", "kind",
        "records", "claim_total", "graded_total", "V+", "V-", "U", "N/A",
        "v_pos_rate", "v_neg_rate", "u_rate", "na_rate_all_claims",
    ]
    delta_fields = [
        "method", "method_name", "family", "group", "group_value",
        "records", "records_with_targets", "status_counts",
        "delta_targets", "delta_recovered", "delta_coverage_rate",
        "sign_matches", "sign_match_rate_all", "sign_match_rate_recovered",
        "magnitude_n", "magnitude_mae", "spearman_n", "spearman",
        "scalar_targets", "scalar_recovered", "scalar_coverage_rate",
        "scalar_within_tolerance", "scalar_within_tolerance_rate", "scalar_mae",
    ]

    outputs = {}
    for name, rows in faith_tables.items():
        path = os.path.join(args.output_dir, f"faithfulness_{name}.csv")
        write_csv(path, rows, faith_fields)
        outputs[f"faithfulness_{name}"] = path
    for name, rows in delta_tables.items():
        path = os.path.join(args.output_dir, f"delta_recovery_{name}.csv")
        write_csv(path, rows, delta_fields)
        outputs[f"delta_recovery_{name}"] = path

    json_path = os.path.join(args.output_dir, "breakdown_tables.json")
    write_json(json_path, {
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "methods": {m: {k: v for k, v in METHODS[m].items() if k in ("name", "family")}
                    for m in args.methods},
        "faithfulness": faith_tables,
        "delta_recovery": delta_tables,
        "outputs": outputs,
    })
    outputs["json"] = json_path

    print("=== Breakdown Tables ===")
    print(f"methods: {', '.join(args.methods)}")
    print("Faithfulness rows:")
    for name, rows in faith_tables.items():
        print(f"  {name:>16}: {len(rows)}")
    print("Delta-Recovery rows:")
    for name, rows in delta_tables.items():
        print(f"  {name:>16}: {len(rows)}")
    for label, path in outputs.items():
        print(f"{label:>28} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
