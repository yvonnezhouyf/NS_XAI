#!/usr/bin/env python3
"""
Generate A4 fixed-initial-reference ablation records.

A4 keeps the full non-stationary input channels:
  - Algorithm State
  - PCTL Analysis Results
  - Derived Metrics Summary

The only intervention is the reference policy. Instead of comparing each
current decision against the adaptive/reference version selected by the main
pipeline, every query in a scenario uses the initial reference model/version
as E_t_prev.

Important distinction:
  - The reference *model version* is fixed to the scenario's initial version.
  - The current request/state remains query-specific. We do not reuse one
    initial MDP tree for every request, because that would mismatch the query's
    epoch, vehicle targets, and operational state.

This script lives entirely under evaluation/ and does not modify the production
XAI pipeline. It patches the loaded scenario data in memory before invoking the
existing orchestrator pipeline.
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ABLATIONS_DIR = os.path.dirname(SCRIPT_DIR)
EVAL_DIR = os.path.dirname(ABLATIONS_DIR)
REPO_DIR = os.path.dirname(EVAL_DIR)
FAITHFULNESS_DIR = os.path.join(EVAL_DIR, "faithfulness")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, FAITHFULNESS_DIR)

from generate_evidence import (  # noqa: E402
    DEFAULT_CSV,
    SCENARIO_LOADERS,
    _build_evidence,
    _force_classifier,
    _to_jsonable,
    filter_rows,
    load_rows,
)

DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "evidence_240.json")
DEFAULT_MDP_TN_CACHE_DIR = os.path.join(SCRIPT_DIR, "mdp_tn_cache")
DEFAULT_MODEL = "gpt-5.4-mini"

A4_FIXED_REFERENCE_PROMPT = """You are generating an operations-facing dispatch explanation for a paratransit service management expert.

Audience:
- The reader has no background in computer science or formal methods.
- Write like an operational dispatch justification, not a technical explanation.
- Keep the explanation concise and practical, like a dispatcher explaining a decision.

Inputs:
- Query
- Algorithm State
- Reference Policy
- PCTL Analysis Results
- Derived Metrics Summary

A4 fixed-reference setting:
- The reference values are from the scenario's initial reference model/version.
- Do not describe the reference as the immediately previous update.
- When explaining a change, describe it as current estimates compared with the initial reference.
- If a metric has only a current value, do not invent an initial-reference value or a temporal change.

Main rules:
- Answer the actual query directly.
- Shape the explanation around the asked query.
- Focus on the 1 to 2 most important reasons only; do not try to explain everything.
- Keep the explanation tight and avoid repeating similar points.
- If the query contains an incorrect or inconsistent assumption about the assignment, explicitly point out the mistake and clarify the correct situation before answering.

Structure guidance:
- Before answering the query, use Algorithm State and Reference Policy to make clear whether the environment has changed, what phase the algorithm is in, whether the current decision is confident, and that comparisons are against the initial reference.
- Use up to three paragraphs that best answer the query.
- Give each paragraph a short bold heading that fits the content of that paragraph.
- Include an alternative-vehicle paragraph only if the query asks for that comparison or the evidence makes that contrast necessary.
- Return the final explanation in GitHub-flavored Markdown.
- At least one paragraph should explain change over time using concrete current-versus-initial system-estimated time or risk comparisons when available.
- Be concise, natural, and operations-facing.
- Anyone with no formal-methods background should be able to understand every sentence.

Change-over-time guidance:
- Use phrases like: the system now estimates..., compared with the initial reference, the system estimated..., relative to the initial reference...
- Make clear these are system estimates, not realized outcomes.
- The main over-time comparison should usually be about the assigned vehicle under current estimates versus the initial reference.
- Only bring in another vehicle if that comparison is needed to answer the query well.

Operational language:
- Do not use MDP_t, MDP_{t-n}, non-stationary, cross-snapshot, snapshot, PCTL, formulas, BNN, or other system-internal method names.
- Do not say "the model thinks," "the AI thinks," or "the planner's understanding."
- It is acceptable to say "the system currently estimates" or "the initial reference estimated" when referring to ETA, travel time, or delay-risk values.
- Use plain operational language such as:
  - the environment has changed / the environment has not changed
  - stable phase / adapting after the change / stable again after the change
  - chance of delay / risk of delay
  - expected passenger wait
  - expected trip completion
  - can begin serving sooner

Probability rules:
- Convert probabilities to percentages with at most one decimal place.
- Prefer pickup-delay or dropoff-delay risk.
- If a probability is at or near saturation (e.g., >=99% or <=1%), express it qualitatively, such as "near certainty" or "almost no risk", instead of exact percentages; when multiple options fall in this range, emphasize relative differences rather than treating them as identical.
- Tie probability statements to operational meaning whenever possible.
- When a probability reflects a future event, explain it as downstream operational risk after taking this request.

Derived metrics:
- Use derived metrics only to support a key point, not to list all values.
- Translate metrics into operational meaning when possible:
  - pickup ETA = expected passenger wait
  - dropoff ETA = expected trip completion time
  - clear current route time = time before the vehicle can start serving this request
  - deadhead to pickup = repositioning time to the pickup
  - dropoff slack = remaining schedule margin at dropoff
- Lower pickup ETA means shorter wait.
- Lower dropoff ETA means faster trip completion.
- Lower deadhead means the vehicle is closer, but closer does not always mean better service.
- Higher dropoff slack means more schedule margin.

Repetition control:
- If an "Already shown to the user in this request" section is present, the user has already seen those points in earlier turns on the same request.
- Do not unnecessarily repeat detailed metrics or full argument structure from prior turns.
- If the current question is broader than a previous one, answer it directly and self-contained at the appropriate level of abstraction.
- Focus on what is new for the current question."""


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(REPO_DIR, ".env"))


def _redirect_mdp_tn_cache(cache_dir: str) -> str:
    """Redirect MDP_tn rebuild pkl cache for this evaluation process only."""
    from core import formula_evaluator

    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    formula_evaluator._MDP_TN_CACHE_DIR = cache_dir
    formula_evaluator.clear_runtime_caches()
    return cache_dir


def _embedded_prompt() -> Tuple[str, str]:
    text = A4_FIXED_REFERENCE_PROMPT.strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, digest


def _load_existing(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"existing output {path} is not a JSON array")
    return data


def _flush(path: str, records: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _version_start_epoch(mcts_data: Dict[str, Any],
                         fixed_version_id: int) -> Optional[int]:
    scenario_data = mcts_data.get("scenario_data", {}) or {}
    version_history = scenario_data.get("model_version_history", {}) or {}
    info = version_history.get(fixed_version_id)
    if isinstance(info, dict):
        return info.get("start_epoch")
    return None


def apply_fixed_reference_policy(mcts_data: Dict[str, Any],
                                 fixed_version_id: int) -> Dict[str, Any]:
    """Patch loaded scenario data in memory so all comparison epochs use the
    same initial reference version as E_t_prev.
    """
    scenario_data = mcts_data.get("scenario_data", {}) or {}
    version_history = scenario_data.get("model_version_history", {}) or {}
    if fixed_version_id not in version_history:
        raise ValueError(
            f"fixed version {fixed_version_id} is not present in "
            "scenario_data.model_version_history"
        )

    comparisons = scenario_data.get("mdp_comparisons", {}) or {}
    patched_epochs = []
    for epoch, comparison in comparisons.items():
        if not isinstance(comparison, dict):
            continue
        state_snapshot = comparison.get("state_snapshot")
        if not isinstance(state_snapshot, dict):
            continue
        original_prev = state_snapshot.get("prev_version_id")
        state_snapshot["prev_version_id"] = fixed_version_id
        comparison.pop("mdp_t_minus_n_tree", None)
        patched_epochs.append({
            "epoch": int(epoch),
            "original_prev_version_id": original_prev,
            "fixed_prev_version_id": fixed_version_id,
        })

    # Formula caches are process-local and some in-memory rebuilt-tree cache is
    # not version-scoped inside scenario_data. Clear defensively after patching.
    try:
        from core.formula_evaluator import clear_runtime_caches
        clear_runtime_caches()
    except Exception:
        pass

    fixed_epoch = _version_start_epoch(mcts_data, fixed_version_id)
    return {
        "reference_policy": "fixed_initial_reference",
        "fixed_reference_version_id": fixed_version_id,
        "fixed_reference_epoch": fixed_epoch,
        "patched_comparison_epochs": len(patched_epochs),
        "patched_epochs": patched_epochs,
    }


def _reference_note(fixed_version_id: int,
                    fixed_epoch: Optional[int]) -> List[str]:
    epoch_text = "unknown epoch" if fixed_epoch is None else f"epoch {fixed_epoch}"
    return [
        "Reference Policy:",
        (
            "  Reference baseline: initial scenario reference "
            f"(version {fixed_version_id}, {epoch_text})."
        ),
        (
            "  Reference values compare the current state against that initial "
            "reference, not against the immediately previous update."
        ),
    ]


def _rewrite_llm_input_for_initial_reference(
    llm_input: str,
    fixed_version_id: int,
    fixed_epoch: Optional[int],
) -> str:
    """Make the full-pipeline input text explicit about the A4 reference."""
    text = llm_input or ""

    # Replace internal labels with human-readable fixed-reference labels.
    version_label = f"Initial reference model (version {fixed_version_id})"
    text = text.replace("MDP_t (current model)", "Current model")
    text = text.replace(f"MDP_{{t-n}} (version {fixed_version_id})", version_label)
    text = re.sub(r"MDP_\{t-n\}(?: \(version [^)]+\))?",
                  version_label, text)
    text = re.sub(r"\bt-n=", "initial=", text)
    text = re.sub(r"\bt=", "current=", text)

    note = "\n".join(_reference_note(fixed_version_id, fixed_epoch))
    marker = "\nPCTL Analysis Results:"
    if marker in text and "Reference Policy:" not in text:
        text = text.replace(marker, f"\n{note}\n{marker}", 1)
    elif "Reference Policy:" not in text:
        text = text.rstrip() + "\n\n" + note + "\n"
    return text


def _call_explanation_llm(llm_input: str, prompt_text: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": llm_input},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def process_query_fixed_reference(
    orchestrator,
    query: str,
    mcts_data: Dict[str, Any],
    *,
    skip_llm: bool,
    force_type: Optional[int],
    fixed_version_id: int,
    fixed_epoch: Optional[int],
    prompt_text: str,
    prompt_sha256: str,
    model: str,
    mdp_tn_cache_dir: str,
) -> Dict[str, Any]:
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

        evidence = _build_evidence(result.get("pctl_results", []),
                                   top_level_epoch=epoch)
        evidence.setdefault("snapshot_meta", {})
        evidence["snapshot_meta"].update({
            "reference_policy": "fixed_initial_reference",
            "fixed_reference_version_id": fixed_version_id,
            "fixed_reference_epoch": fixed_epoch,
        })

        llm_input = orchestrator._build_explanation_prompt(
            query,
            result,
            env_context,
            tree_node,
            id_map=id_map,
            mcts_data=mcts_data,
            epoch=epoch,
            vehicle_targets=resolved_targets,
        )
        llm_input = _rewrite_llm_input_for_initial_reference(
            llm_input, fixed_version_id, fixed_epoch
        )

        rec: Dict[str, Any] = {
            "epoch": epoch,
            "classification_label": result.get("classification_label"),
            "pctl_formulas": result.get("pctl_formulas"),
            "resolved_vehicle_targets": _to_jsonable(resolved_targets),
            "evidence": _to_jsonable(evidence),
            "llm_input": llm_input,
            "ablation_meta": {
                "ablation_id": "A4",
                "name": "fixed-initial-reference",
                "reference_policy": (
                    "fixed initial scenario reference model/version; "
                    "current state remains query-specific"
                ),
                "fixed_reference_version_id": fixed_version_id,
                "fixed_reference_epoch": fixed_epoch,
                "mdp_tn_cache_dir": mdp_tn_cache_dir,
                "prompt_source": "embedded:A4_FIXED_REFERENCE_PROMPT",
                "prompt_sha256": prompt_sha256,
                "model": model,
                "input_group_policy": {
                    "algorithm_state": "kept",
                    "pctl_analysis_results": (
                        "current values plus fixed initial reference values"
                    ),
                    "derived_metrics_summary": (
                        "current values plus fixed initial reference values"
                    ),
                },
            },
        }

        if skip_llm:
            rec["explanation"] = None
        else:
            rec["explanation"] = _call_explanation_llm(
                llm_input, prompt_text, model
            )
        return rec
    finally:
        if restore is not None:
            restore()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=DEFAULT_CSV,
                   help="Path to core_queries_240.csv")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Output JSON path")
    p.add_argument("--mdp-tn-cache-dir", default=DEFAULT_MDP_TN_CACHE_DIR,
                   help="A4-local cache for rebuilt fixed-reference MDP trees")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--fixed-version-id", type=int, default=0,
                   help="Scenario model version to use as fixed reference")
    p.add_argument("--types", nargs="+", type=int)
    p.add_argument("--scenarios", nargs="+", type=int, choices=[0, 1, 2])
    p.add_argument("--query-ids", nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--interactive-cache", action="store_true",
                   help="Allow interactive MDP_tn cache prompts")
    p.add_argument("--force-type-from-csv", action="store_true",
                   help="Bypass query classification and use CSV type_k")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _load_dotenv()

    if not args.interactive_cache:
        os.environ.setdefault("NS_XAI_NON_INTERACTIVE", "1")

    mdp_tn_cache_dir = _redirect_mdp_tn_cache(args.mdp_tn_cache_dir)

    rows = filter_rows(
        load_rows(args.csv),
        args.types,
        args.scenarios,
        args.query_ids,
        args.limit,
    )
    if not rows:
        print("No rows match filters; nothing to do.")
        return 0

    existing_records: List[Dict[str, Any]] = []
    done_ids = set()
    if args.resume and os.path.exists(args.output):
        existing_records = _load_existing(args.output)
        done_ids = {
            r.get("query_id") for r in existing_records
            if isinstance(r, dict)
        }
        rows = [r for r in rows if r.get("query_id") not in done_ids]
        print(f"[resume] {len(done_ids)} already done; {len(rows)} remaining")

    if not args.resume:
        existing_records = []

    if not rows:
        print("Nothing to do.")
        return 0

    rows_by_scenario: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        rows_by_scenario.setdefault(int(row["scenario"]), []).append(row)

    prompt_text, prompt_sha256 = _embedded_prompt()

    print("Initializing orchestrator...")
    from use_cases.paratransit.config import ParatransitConfig
    from core.orchestrator import NSXAIOrchestrator
    import main_paratransit_demo as demo

    orchestrator = NSXAIOrchestrator(ParatransitConfig())
    all_records: List[Dict[str, Any]] = list(existing_records)
    n_ok = n_err = 0
    t_start = time.time()

    print(
        f"Generating A4 fixed-initial-reference records -> {args.output}\n"
        f"rows={len(rows)} fixed_version={args.fixed_version_id} "
        f"skip_llm={args.skip_llm} model={args.model}\n"
        f"mdp_tn_cache_dir={mdp_tn_cache_dir}"
    )

    for scenario_id, scenario_rows in sorted(rows_by_scenario.items()):
        loader_name = SCENARIO_LOADERS.get(scenario_id)
        if loader_name is None:
            print(f"[skip] scenario {scenario_id}: no loader registered")
            continue

        print(f"\n=== Scenario {scenario_id} ({loader_name}): "
              f"{len(scenario_rows)} queries ===")
        mcts_data = getattr(demo, loader_name)(use_saved=True)
        if mcts_data is None:
            print(f"[error] failed to load scenario {scenario_id}")
            continue

        reference_meta = apply_fixed_reference_policy(
            mcts_data, args.fixed_version_id
        )
        fixed_epoch = reference_meta.get("fixed_reference_epoch")
        print(
            "  Fixed reference: "
            f"version={args.fixed_version_id}, epoch={fixed_epoch}, "
            f"patched_epochs={reference_meta['patched_comparison_epochs']}"
        )

        for i, row in enumerate(scenario_rows, 1):
            qid = row["query_id"]
            qtxt = row["query_text"]
            type_k = int(row["type_k"])
            t_query = time.time()
            print(f"  [{i}/{len(scenario_rows)}] {qid} "
                  f"(type {type_k}) - {qtxt[:80]}")

            rec: Dict[str, Any] = {
                "query_id": qid,
                "type_k": type_k,
                "scenario": scenario_id,
                "query_text": qtxt,
                "timestamp": (
                    datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z")
                ),
            }

            try:
                forced = type_k if args.force_type_from_csv else None
                rec.update(process_query_fixed_reference(
                    orchestrator,
                    qtxt,
                    mcts_data,
                    skip_llm=args.skip_llm,
                    force_type=forced,
                    fixed_version_id=args.fixed_version_id,
                    fixed_epoch=fixed_epoch,
                    prompt_text=prompt_text,
                    prompt_sha256=prompt_sha256,
                    model=args.model,
                    mdp_tn_cache_dir=mdp_tn_cache_dir,
                ))
                rec["status"] = "ok"
                n_ok += 1
            except Exception:
                rec["status"] = "error"
                rec["error"] = traceback.format_exc()
                rec["explanation"] = None
                n_err += 1
                print(f"    ERROR:\n{rec['error']}")

            rec["elapsed_sec"] = round(time.time() - t_query, 2)
            all_records.append(rec)
            _flush(args.output, all_records)

    elapsed = time.time() - t_start
    print(f"\nDone. ok={n_ok} err={n_err} elapsed={elapsed:.1f}s -> "
          f"{args.output}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
