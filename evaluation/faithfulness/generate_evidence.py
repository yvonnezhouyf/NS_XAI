#!/usr/bin/env python3
"""
Generate per-query evidence + non-stationary explanation records for
faithfulness evaluation.

For each query in core_queries_240.csv, this script:
  1. Loads the right scenario's mcts_data (case 0/1/2 cached across queries).
  2. Runs the orchestrator pipeline to get PCTL evidence (E_t, E_t_prev).
  3. Generates the non-stationary explanation (the only one we evaluate).
  4. Appends the record to a single pretty-printed JSON array at --output.

Per-record schema:
  {
    "query_id", "type_k", "scenario", "query_text",
    "timestamp", "epoch", "classification_label",
    "resolved_vehicle_targets",
    "pctl_formulas",                      # query-specific list (count varies by type)
    "evidence": {
      "E_t":          {formula: {action: value}},
      "E_t_prev":     {formula: {action: value}},
      "snapshot_meta": {prev_version_id, prev_epoch, is_compare_epoch,
                        event_info, counter_intuitive_info}
    },
    "llm_input",                          # the NS prompt actually sent
    "explanation",                        # the NS explanation text
    "status", "elapsed_sec"
  }

Usage:
    conda activate xai39
    cd .

    # Smoke test: just 3 queries, no LLM calls
    python faithfulness/generate_evidence.py --limit 3 --skip-llm \
        --output faithfulness/evidence_smoke.json

    # Pilot: 1 type, 1 scenario with LLM
    python faithfulness/generate_evidence.py --types 1 --scenarios 1 \
        --output faithfulness/evidence_pilot.json

    # Full run, resumable
    python faithfulness/generate_evidence.py \
        --output faithfulness/evidence_240.json --resume
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../evaluation/faithfulness
EVAL_DIR = os.path.dirname(SCRIPT_DIR)                     # .../evaluation
REPO_DIR = os.path.dirname(EVAL_DIR)                       # .../ns_explainer
sys.path.insert(0, REPO_DIR)

DEFAULT_CSV = os.path.join(EVAL_DIR, "core_queries_240.csv")

# CSV scenario id -> demo loader name
SCENARIO_LOADERS = {
    0: "load_scenario0_final_data",
    1: "load_counter_intuitive_data",
    2: "load_event_scenario_data",
}


def _to_jsonable(obj: Any) -> Any:
    """Best-effort conversion of nested dict/list/scalars to JSON-safe types."""
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj
    if isinstance(obj, float):
        return obj if obj == obj else None  # NaN -> None
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    # numpy scalars / arrays
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def _extract_action_value(action_entry: Any) -> Any:
    """Pull the headline numeric value from an action's evidence dict."""
    if isinstance(action_entry, dict):
        for k in ("pctl_value", "value", "result"):
            if k in action_entry:
                return action_entry[k]
        return action_entry
    return action_entry


def _build_evidence(pctl_results: List[Dict[str, Any]],
                    top_level_epoch: Optional[int] = None) -> Dict[str, Any]:
    """Flatten orchestrator pctl_results into E_t, E_t_prev, plus a single
    record-level snapshot_meta. The grader can compute E_t - E_t_prev on demand,
    and visit counts are dropped (not needed for claim grading).

    Output shape:
      {
        "E_t":      {formula: {action: value, ...}, ...},
        "E_t_prev": {formula: {action: value, ...}, ...},
        "snapshot_meta": {prev_version_id, prev_epoch, is_compare_epoch,
                          event_info, counter_intuitive_info}
      }
    """
    E_t: Dict[str, Dict[str, Any]] = {}
    E_t_prev: Dict[str, Dict[str, Any]] = {}
    snapshot_meta: Dict[str, Any] = {}

    def _merge_meta(r: Dict[str, Any]) -> None:
        for k in ("prev_version_id", "prev_epoch", "is_compare_epoch",
                  "event_info", "counter_intuitive_info"):
            v = r.get(k)
            if v is not None and snapshot_meta.get(k) is None:
                snapshot_meta[k] = v

    for item in pctl_results or []:
        if "formula" not in item or "result" not in item:
            continue
        f = item["formula"]
        r = item["result"] or {}
        mdp_t = r.get("mdp_t") or {}
        mdp_tn = r.get("mdp_t_minus_n") or {}

        cur = {a: _extract_action_value(v) for a, v in mdp_t.items()} if isinstance(mdp_t, dict) else mdp_t
        prv = {a: _extract_action_value(v) for a, v in mdp_tn.items()} if isinstance(mdp_tn, dict) else mdp_tn

        E_t[f] = cur
        E_t_prev[f] = prv

        _merge_meta(r)

    # Drop counter_intuitive_info.decision_epoch (== top-level epoch).
    ci = snapshot_meta.get("counter_intuitive_info")
    if isinstance(ci, dict) and ci.get("decision_epoch") == top_level_epoch:
        ci = {k: v for k, v in ci.items() if k != "decision_epoch"}
        snapshot_meta["counter_intuitive_info"] = ci

    return {
        "E_t": _to_jsonable(E_t),
        "E_t_prev": _to_jsonable(E_t_prev),
        "snapshot_meta": _to_jsonable(snapshot_meta),
    }


def _run_ns_llm(ns_input: str, ns_pid: str) -> str:
    from core.explanation_generation import generate_explanation_async

    async def _gather():
        return await generate_explanation_async(ns_input, ns_pid)

    return asyncio.run(_gather())


def _force_classifier(forced_type: int, level: str = "MICRO"):
    """Monkey-patch the classifier symbol that orchestrator imports so that
    `_run_query_pipeline` -> `process_query` reads `forced_type` instead of
    calling the LLM.

    Returns a (install, restore) pair. The orchestrator's own import binding is
    what `process_query` actually reads, so we patch on `core.orchestrator`,
    not on `core.query_classification`.
    """
    import core.orchestrator as _orch_mod
    original = _orch_mod.classify_query_with_openai

    def _patched(query, prompt_id):  # signature must match real classifier
        return forced_type, level

    def install():
        _orch_mod.classify_query_with_openai = _patched

    def restore():
        _orch_mod.classify_query_with_openai = original

    return install, restore


def process_query(orchestrator, query: str, mcts_data: Dict, *,
                  skip_llm: bool,
                  force_type: Optional[int] = None) -> Dict[str, Any]:
    """Run pipeline + (optionally) LLM for one query. Returns a JSON-safe record fragment.

    Only the non-stationary path is captured — faithfulness eval targets the
    NS explanation, and Ablations 1/2/3 rebuild their prompts from `evidence`
    rather than from a stored stationary baseline.

    If `force_type` is given, the LLM classifier is bypassed and that type_id
    is used directly (saves an API call and pins Spec(q) construction).
    """
    restore = None
    if force_type is not None:
        install, restore = _force_classifier(force_type)
        install()
    try:
        pipe = orchestrator._run_query_pipeline(query, mcts_data)
        result = pipe["result"]
        epoch = pipe["epoch"]
        tree_node = pipe["tree_node"]
        env_context = pipe["env_context"]
        id_map = pipe["id_map"]
        resolved_targets = pipe.get("resolved_targets", [])

        evidence = _build_evidence(result.get("pctl_results", []), top_level_epoch=epoch)

        record: Dict[str, Any] = {
            "epoch": epoch,
            "classification_label": result.get("classification_label"),
            "pctl_formulas": result.get("pctl_formulas"),
            "resolved_vehicle_targets": _to_jsonable(resolved_targets),
            "evidence": evidence,
        }

        ns_input = orchestrator._build_explanation_prompt(
            query, result, env_context, tree_node,
            id_map=id_map, mcts_data=mcts_data, epoch=epoch,
            vehicle_targets=resolved_targets)
        record["llm_input"] = ns_input

        if skip_llm:
            record["explanation"] = None
        else:
            ns_pid = orchestrator.config.EXPLANATION_GENERATION_PROMPT_ID
            record["explanation"] = _run_ns_llm(ns_input, ns_pid)

        return record
    finally:
        if restore is not None:
            restore()


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def filter_rows(rows, types, scenarios, query_ids, limit):
    out = rows
    if types:
        types_set = {str(t) for t in types}
        out = [r for r in out if r["type_k"] in types_set]
    if scenarios:
        sc_set = {str(s) for s in scenarios}
        out = [r for r in out if r["scenario"] in sc_set]
    if query_ids:
        qid_set = set(query_ids)
        out = [r for r in out if r["query_id"] in qid_set]
    if limit is not None:
        out = out[:limit]
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=DEFAULT_CSV, help="Path to core_queries_240.csv")
    p.add_argument("--output", required=True,
                   help="Output JSON file (single pretty-printed array of records).")
    p.add_argument("--types", nargs="+", type=int, help="Filter by type_k (e.g. --types 1 2)")
    p.add_argument("--scenarios", nargs="+", type=int, choices=[0, 1, 2],
                   help="Filter by scenario id")
    p.add_argument("--query-ids", nargs="+", help="Run only these specific query_ids")
    p.add_argument("--limit", type=int, help="Cap number of queries (after filtering)")
    p.add_argument("--skip-llm", action="store_true",
                   help="Skip LLM explanation generation (evidence + prompts only)")
    p.add_argument("--resume", action="store_true",
                   help="Skip query_ids already present in --output (parses existing array).")
    p.add_argument("--interactive-cache", action="store_true",
                   help="Allow interactive 'Load from cache?' prompts. "
                        "By default we set NS_XAI_NON_INTERACTIVE=1, which "
                        "auto-accepts the cache (identical to pressing Enter).")
    p.add_argument("--force-type-from-csv", action="store_true",
                   help="Bypass the LLM classifier and use type_k from "
                        "core_queries_240.csv directly. Useful for queries the "
                        "classifier intermittently labels as -1 — pinning the "
                        "gold type avoids contaminating Spec(q) construction.")
    return p.parse_args()


def main():
    args = parse_args()

    # Auto-accept the cache prompt unless user explicitly opts in to interactive.
    # The flag is read by core/formula_evaluator.py at the prompt site; default
    # branch (use_cached=True) is identical to pressing Enter at the prompt,
    # so output is bit-for-bit equivalent to the interactive run.
    if not args.interactive_cache:
        os.environ.setdefault("NS_XAI_NON_INTERACTIVE", "1")

    rows = filter_rows(load_rows(args.csv),
                       args.types, args.scenarios, args.query_ids, args.limit)

    if not rows:
        print("No rows match filters; nothing to do.")
        return 0

    # Load existing records (for resume + to preserve previously-completed entries
    # when we rewrite the output array after each new record).
    existing_records: List[Dict[str, Any]] = []
    done_ids: set = set()
    if os.path.exists(args.output):
        try:
            with open(args.output) as f:
                existing_records = json.load(f)
            if not isinstance(existing_records, list):
                raise ValueError("existing output is not a JSON array")
        except Exception as e:
            print(f"[warn] could not parse existing {args.output} ({e}); starting fresh")
            existing_records = []

    if args.resume:
        done_ids = {r.get("query_id") for r in existing_records if isinstance(r, dict)}
        rows = [r for r in rows if r["query_id"] not in done_ids]
        print(f"[resume] {len(done_ids)} already done; {len(rows)} remaining")
    else:
        # Not resuming: discard any old content.
        existing_records = []

    # Group by scenario so each scenario's mcts_data is loaded once.
    rows_by_scenario: Dict[int, List[Dict[str, str]]] = {}
    for r in rows:
        rows_by_scenario.setdefault(int(r["scenario"]), []).append(r)

    # Init orchestrator (heavy import — do once)
    print("Initializing orchestrator...")
    from use_cases.paratransit.config import ParatransitConfig
    from core.orchestrator import NSXAIOrchestrator
    import main_paratransit_demo as demo
    orchestrator = NSXAIOrchestrator(ParatransitConfig())

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    n_ok = n_err = 0
    t0 = time.time()

    all_records: List[Dict[str, Any]] = list(existing_records)

    def _flush() -> None:
        # Atomic-ish write: write to tmp then rename, so a crash mid-flush
        # doesn't corrupt the previously-good output.
        tmp = args.output + ".tmp"
        with open(tmp, "w") as out:
            json.dump(all_records, out, indent=2, ensure_ascii=False)
        os.replace(tmp, args.output)

    for sc_id, sc_rows in sorted(rows_by_scenario.items()):
        loader_name = SCENARIO_LOADERS.get(sc_id)
        if loader_name is None:
            print(f"[skip] scenario {sc_id}: no loader registered")
            continue

        print(f"\n=== Scenario {sc_id} ({loader_name}): {len(sc_rows)} queries ===")
        mcts_data = getattr(demo, loader_name)(use_saved=True)
        if mcts_data is None:
            print(f"[error] failed to load scenario {sc_id}")
            continue

        for i, row in enumerate(sc_rows, 1):
            qid = row["query_id"]
            qtxt = row["query_text"]
            tk = row["type_k"]
            t_q = time.time()
            print(f"  [{i}/{len(sc_rows)}] {qid} (type {tk}) — {qtxt[:80]}")

            rec: Dict[str, Any] = {
                "query_id": qid,
                "type_k": int(tk),
                "scenario": sc_id,
                "query_text": qtxt,
                "timestamp": datetime.utcnow().isoformat() + "Z",
            }
            try:
                forced = int(tk) if args.force_type_from_csv else None
                rec.update(process_query(orchestrator, qtxt, mcts_data,
                                         skip_llm=args.skip_llm,
                                         force_type=forced))
                rec["status"] = "ok"
                n_ok += 1
            except Exception:
                rec["status"] = "error"
                rec["error"] = traceback.format_exc()
                n_err += 1
                print(f"    ERROR:\n{rec['error']}")

            rec["elapsed_sec"] = round(time.time() - t_q, 2)
            all_records.append(rec)
            _flush()

    dt = time.time() - t0
    print(f"\nDone. ok={n_ok} err={n_err} elapsed={dt:.1f}s -> {args.output}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
