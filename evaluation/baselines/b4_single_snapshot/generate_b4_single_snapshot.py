#!/usr/bin/env python3
"""
Generate B4 / A1 single-snapshot explanation records.

B4 keeps the same 240 benchmark queries and the same downstream
claim-extraction / claim-verification pipeline as the main faithfulness run,
but it constructs a clean single-snapshot LLM input from structured evidence:
  - PCTL Analysis Results contain current-snapshot values only.
  - Derived Metrics Summary contains current-snapshot values only.
  - Algorithm State is treated as derived; only current planner confidence is
    kept, while environment-change status and algorithm phase are omitted.
  - Previous/reference-snapshot values, deltas, recovery intervals, and
    turning intervals are not present in the input.

The generator uses an embedded single-snapshot prompt derived from ns_prompt.md,
not the full non-stationary stored prompt id. The output schema intentionally
matches evaluation/faithfulness/evidence_240.json so the existing extractor and
verifier can be reused with --input / --evidence pointing to the B4 output file.
"""

import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINES_DIR = os.path.dirname(SCRIPT_DIR)
EVAL_DIR = os.path.dirname(BASELINES_DIR)
REPO_DIR = os.path.dirname(EVAL_DIR)
sys.path.insert(0, REPO_DIR)

DEFAULT_INPUT = os.path.join(EVAL_DIR, "faithfulness", "evidence_240.json")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "evidence_240.json")
DEFAULT_MODEL = "gpt-5.4-mini"

SINGLE_SNAPSHOT_PROMPT = """You are generating an operations-facing dispatch explanation for a paratransit service management expert.

Audience:
- The reader has no background in computer science or formal methods.
- Write like an operational dispatch justification, not a technical explanation.
- Keep the explanation concise and practical, like a dispatcher explaining a decision.

Inputs:
- Query
- Algorithm State
- PCTL Analysis Results
- Derived Metrics Summary

Single-snapshot setting:
- The input contains only the current dispatch snapshot.
- Do not infer or describe previous/reference-snapshot values.
- Do not describe temporal deltas, recovery intervals, turning intervals, or before/after comparisons.
- If the query asks about how something changed over time, answer that the single-snapshot evidence cannot determine the temporal change, then summarize what the current evidence does show.
- Algorithm State is part of the derived-metrics channel. In this setting, it may contain current planner confidence only; do not infer environment-change status or algorithm phase unless explicitly provided.

Main rules:
- Answer the actual query directly.
- Shape the explanation around the asked query.
- Focus on the 1 to 2 most important current-snapshot reasons only; do not try to explain everything.
- Keep the explanation tight and avoid repeating similar points.
- If the query contains an incorrect or inconsistent assumption about the assignment, explicitly point out the mistake and clarify the correct current situation before answering.

Structure guidance:
- Use up to three paragraphs that best answer the query.
- Give each paragraph a short bold heading that fits the content of that paragraph.
- Include an alternative-vehicle paragraph only if the query asks for that comparison or the current evidence makes that contrast necessary.
- Return the final explanation in GitHub-flavored Markdown.
- Do not include a change-over-time paragraph unless the input explicitly contains previous/reference evidence. In this single-snapshot ablation, it normally will not.
- Be concise, natural, and operations-facing.
- Anyone with no formal-methods background should be able to understand every sentence.

Operational language:
- Do not use MDP_t, MDP_{t-n}, non-stationary, cross-snapshot, snapshot, PCTL, formulas, BNN, or other system-internal method names.
- Do not say "the model thinks," "the AI thinks," or "the planner's understanding."
- It is acceptable to say "the system currently estimates" when referring to ETA, travel time, or delay-risk values.
- Use plain operational language such as:
  - current dispatch state
  - current confidence
  - chance of delay / risk of delay
  - expected passenger wait
  - expected trip completion
  - can begin serving sooner

Probability rules:
- Convert probabilities to percentages with at most one decimal place.
- Prefer pickup-delay or dropoff-delay risk.
- If a probability is at or near saturation (e.g., >=99% or <=1%), express it qualitatively (e.g., "near certainty" or "almost no risk") instead of exact percentages; when multiple options fall in this range, emphasize relative differences rather than treating them as identical, and avoid mixing qualitative labels with precise extreme values.
- Tie probability statements to operational meaning whenever possible.
- When a probability reflects a future event, explain it as the downstream operational risk after taking this request. Do not describe it as happening strictly during the execution of the current request.

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

TEMPORAL_ONLY_METRICS = {
    "DERIVED: TURNING_INTERVAL",
}

PCTL_SECTION_HEADER = "PCTL Analysis Results:"
DERIVED_SECTION_HEADER = "Derived Metrics Summary:"

DERIVED_GROUPS: List[Tuple[str, List[str]]] = [
    ("System Estimated Travel-Time", [
        "DERIVED: ETA_PICKUP_BY_VEHICLE",
        "DERIVED: ETA_DROPOFF_BY_VEHICLE",
    ]),
    ("Workload summary", [
        "DERIVED: PENDING_REQUESTS_BY_VEHICLE",
    ]),
    ("Route burden summary", [
        "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE",
        "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE",
        "DERIVED: DROPOFF_SLACK_BY_VEHICLE",
        "DERIVED: SERVICE_PATH_DISRUPTION_BY_VEHICLE",
    ]),
    ("Global context summary", [
        "DERIVED: TRAFFIC_LEVEL",
    ]),
]

METRIC_UNITS = {
    "DERIVED: ETA_PICKUP_BY_VEHICLE": "minutes",
    "DERIVED: ETA_DROPOFF_BY_VEHICLE": "minutes",
    "DERIVED: CLEAR_CURRENT_ROUTE_TIME_BY_VEHICLE": "minutes",
    "DERIVED: DEADHEAD_TO_PICKUP_BY_VEHICLE": "minutes",
    "DERIVED: DROPOFF_SLACK_BY_VEHICLE": "minutes",
}


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(REPO_DIR, ".env"))


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return str(obj)


def _load_records(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array")
    return records


def _embedded_prompt() -> Tuple[str, str]:
    text = SINGLE_SNAPSHOT_PROMPT.strip()
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, digest


def _filter_records(records: List[Dict[str, Any]], query_ids: Optional[List[str]],
                    limit: Optional[int]) -> List[Dict[str, Any]]:
    out = records
    if query_ids:
        qid_set = set(query_ids)
        out = [r for r in out if r.get("query_id") in qid_set]
    if limit is not None:
        out = out[:limit]
    return out


def _load_existing(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"existing output {path} is not a JSON array")
    return data


def _ordered(records_by_id: Dict[str, Dict[str, Any]],
             order: Iterable[str]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for qid in order:
        if qid in records_by_id:
            out.append(records_by_id[qid])
            seen.add(qid)
    for qid in sorted(k for k in records_by_id if k not in seen):
        out.append(records_by_id[qid])
    return out


def _flush(path: str, records_by_id: Dict[str, Dict[str, Any]],
           order: Iterable[str]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(_ordered(records_by_id, order), f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _vehicle_sort_key(item: Tuple[str, Any]) -> Tuple[int, str]:
    key, _ = item
    if isinstance(key, str) and re.fullmatch(r"V\d+", key):
        return (int(key[1:]), key)
    return (999, str(key))


def _format_number(value: float, *, pctl: bool = False) -> str:
    if pctl:
        return f"{value:.4f}"
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}"


def _format_value(value: Any, *, pctl: bool = False) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (int, float)):
        return _format_number(float(value), pctl=pctl)
    return str(value)


def _filter_metric_value(value: Any, keys: Optional[Sequence[str]]) -> Any:
    if not keys or not isinstance(value, dict):
        return value
    return {k: value[k] for k in keys if k in value}


def _format_metric_values(value: Any, *, pctl: bool = False,
                          keys: Optional[Sequence[str]] = None) -> str:
    value = _filter_metric_value(value, keys)
    if isinstance(value, dict):
        items = sorted(value.items(), key=_vehicle_sort_key)
        return "{" + ", ".join(
            f"{k}={_format_value(v, pctl=pctl)}" for k, v in items
        ) + "}"
    return _format_value(value, pctl=pctl)


def _section_lines(text: str, header: str,
                   stop_headers: Sequence[str] = ()) -> List[str]:
    lines = (text or "").splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == header)
    except StopIteration:
        return []

    out = []
    for line in lines[start + 1:]:
        stripped = line.strip()
        if stripped in stop_headers:
            break
        out.append(line)
    return out


def _vehicle_keys_in_current_side(line: str) -> Optional[List[str]]:
    current_side = line.split("|", 1)[0]
    keys = []
    for m in re.finditer(r"\b(V\d+)\s*=", current_side):
        key = m.group(1)
        if key not in keys:
            keys.append(key)
    return keys or None


def _extract_visible_pctl_specs(source_record: Dict[str, Any]
                                ) -> List[Tuple[str, Optional[List[str]]]]:
    """Preserve the full pipeline's query-specific PCTL compression choice,
    but render values from structured current-snapshot evidence only."""
    evidence = source_record.get("evidence", {}).get("E_t", {}) or {}
    lines = _section_lines(
        source_record.get("llm_input") or "",
        PCTL_SECTION_HEADER,
        stop_headers=(DERIVED_SECTION_HEADER,),
    )
    specs: List[Tuple[str, Optional[List[str]]]] = []
    current_metric: Optional[str] = None
    for line in lines:
        m = re.match(r"^\s{2}(.+):\s*$", line)
        if m:
            metric = m.group(1).strip()
            current_metric = (
                metric if metric in evidence and not metric.startswith("DERIVED:")
                else None
            )
            continue
        if current_metric and "Current model:" in line:
            specs.append((current_metric, _vehicle_keys_in_current_side(line)))
            current_metric = None
        elif current_metric and "MDP_t (current model):" in line:
            specs.append((current_metric, _vehicle_keys_in_current_side(line)))
            current_metric = None

    if specs:
        return specs

    # Fallback for records whose stored prompt lacks a parseable PCTL block.
    return [
        (f, None) for f in (source_record.get("pctl_formulas") or [])
        if f in evidence and not str(f).startswith("DERIVED:")
    ]


def _extract_visible_derived_specs(source_record: Dict[str, Any]
                                   ) -> List[Tuple[str, Optional[List[str]]]]:
    """Preserve the full pipeline's query-specific derived-metric selection,
    excluding temporal-only metrics."""
    evidence = source_record.get("evidence", {}).get("E_t", {}) or {}
    lines = _section_lines(source_record.get("llm_input") or "",
                           DERIVED_SECTION_HEADER)
    specs: List[Tuple[str, Optional[List[str]]]] = []
    for line in lines:
        m = re.match(r"^\s{4}([A-Z0-9_]+)(?:\s+\([^)]+\))?:", line)
        if not m:
            continue
        metric = f"DERIVED: {m.group(1)}"
        if metric in evidence and metric not in TEMPORAL_ONLY_METRICS:
            specs.append((metric, _vehicle_keys_in_current_side(line)))

    if specs:
        return specs

    return [
        (f, None) for f in (source_record.get("pctl_formulas") or [])
        if f in evidence and str(f).startswith("DERIVED:")
        and f not in TEMPORAL_ONLY_METRICS
    ]


def _extract_planner_confidence(source_record: Dict[str, Any]) -> Optional[str]:
    llm_input = source_record.get("llm_input") or ""
    m = re.search(r"^\s*Planner confidence:\s*(.+?)\s*$", llm_input, re.M)
    if m:
        return m.group(1).strip()
    return None


def _derived_display_name(metric: str) -> str:
    return metric.replace("DERIVED: ", "")


def _format_pctl_section(source_record: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    evidence = source_record.get("evidence", {}).get("E_t", {}) or {}
    specs = _extract_visible_pctl_specs(source_record)
    lines = [PCTL_SECTION_HEADER]
    rendered = []

    if not specs:
        lines.append("No current-snapshot PCTL metrics available.")
        return lines, rendered

    for metric, keys in specs:
        if metric not in evidence:
            continue
        lines.append(f"  {metric}:")
        lines.append(
            "    Current model: "
            f"{_format_metric_values(evidence[metric], pctl=True, keys=keys)}"
        )
        rendered.append(metric)

    if not rendered:
        lines.append("No current-snapshot PCTL metrics available.")
    return lines, rendered


def _format_derived_section(source_record: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    evidence = source_record.get("evidence", {}).get("E_t", {}) or {}
    spec_map = {
        metric: keys
        for metric, keys in _extract_visible_derived_specs(source_record)
    }
    lines = [DERIVED_SECTION_HEADER]
    rendered: List[str] = []

    for group_name, metrics in DERIVED_GROUPS:
        group_lines: List[str] = []
        for metric in metrics:
            if metric not in spec_map or metric not in evidence:
                continue
            name = _derived_display_name(metric)
            unit = METRIC_UNITS.get(metric)
            label = f"{name} ({unit})" if unit else name
            group_lines.append(
                f"    {label}: "
                f"{_format_metric_values(evidence[metric], keys=spec_map[metric])}"
            )
            rendered.append(metric)
        if group_lines:
            lines.append(f"  [{group_name}]")
            lines.extend(group_lines)

    if not rendered:
        lines.append("No current-snapshot derived metrics available.")
    return lines, rendered


def build_single_snapshot_input(source_record: Dict[str, Any]
                                ) -> Tuple[str, Dict[str, Any]]:
    """Build a clean B4 input from structured current-snapshot evidence."""
    q = source_record.get("query_text") or ""
    confidence = _extract_planner_confidence(source_record)
    pctl_lines, pctl_metrics = _format_pctl_section(source_record)
    derived_lines, derived_metrics = _format_derived_section(source_record)

    lines: List[str] = [
        f"Query: {q}",
        "",
        "Algorithm State:",
    ]
    if confidence:
        lines.append(f"  Planner confidence: {confidence}")
    else:
        lines.append("  Planner confidence: unavailable")
    lines.extend(["", *pctl_lines, "", *derived_lines, ""])

    meta = {
        "input_builder": "structured_current_snapshot",
        "included_pctl_metrics": pctl_metrics,
        "included_derived_metrics": derived_metrics,
        "algorithm_state_fields": (
            ["Planner confidence"] if confidence else ["Planner confidence unavailable"]
        ),
        "excluded_fields": [
            "E_t_prev",
            "temporal deltas",
            "DERIVED: TURNING_INTERVAL",
            "Environment changed",
            "Algorithm phase",
        ],
    }
    return "\n".join(lines).strip() + "\n", meta


def _allowed_metric_specs(source_record: Dict[str, Any]
                          ) -> Dict[str, Optional[List[str]]]:
    specs: Dict[str, Optional[List[str]]] = {}
    for metric, keys in _extract_visible_pctl_specs(source_record):
        specs[metric] = keys
    for metric, keys in _extract_visible_derived_specs(source_record):
        specs[metric] = keys
    return specs


def _build_b4_evidence(source_record: Dict[str, Any]) -> Dict[str, Any]:
    source_evidence = source_record.get("evidence") or {}
    e_t = source_evidence.get("E_t") or {}
    if not isinstance(e_t, dict):
        e_t = {}

    allowed_specs = _allowed_metric_specs(source_record)
    current_only = {}
    for metric, keys in allowed_specs.items():
        if metric in e_t and metric not in TEMPORAL_ONLY_METRICS:
            current_only[metric] = _filter_metric_value(e_t[metric], keys)

    # E_t_prev is intentionally empty. Accidental previous/delta claims should
    # be unsupported under the B4 evidence scope.
    return {
        "E_t": _jsonable(current_only),
        "E_t_prev": {},
        "snapshot_meta": {
            "baseline": "B4/A1 single-snapshot",
            "source_has_E_t_prev": bool((source_evidence.get("E_t_prev") or {})),
        },
    }


async def _call_explanation_llm(llm_input: str, prompt_text: str,
                                model: str) -> str:
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_text},
            {"role": "user", "content": llm_input},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


async def _build_record(source_record: Dict[str, Any], prompt_text: str,
                        prompt_sha256: str, model: str, skip_llm: bool,
                        timeout_sec: float) -> Dict[str, Any]:
    t0 = time.time()
    qid = source_record.get("query_id")
    rec: Dict[str, Any] = {
        "query_id": qid,
        "type_k": source_record.get("type_k"),
        "scenario": source_record.get("scenario"),
        "query_text": source_record.get("query_text"),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "epoch": source_record.get("epoch"),
        "classification_label": source_record.get("classification_label"),
        "resolved_vehicle_targets": source_record.get("resolved_vehicle_targets", []),
        "pctl_formulas": [
            f for f in _allowed_metric_specs(source_record)
            if f not in TEMPORAL_ONLY_METRICS
        ],
        "evidence": _build_b4_evidence(source_record),
        "baseline_meta": {
            "baseline_id": "B4",
            "ablation_id": "A1",
            "name": "single-snapshot",
            "source_query_id": qid,
            "source_scope": "current snapshot only",
            "prompt_source": "embedded:SINGLE_SNAPSHOT_PROMPT",
            "prompt_sha256": prompt_sha256,
            "model": model,
            "input_group_policy": {
                "pctl_analysis_results": "current snapshot only",
                "algorithm_state": (
                    "treated as derived; keep current planner confidence, "
                    "remove environment changed and algorithm phase"
                ),
                "derived_metrics_summary": (
                    "current snapshot only; remove t-n values and temporal-only "
                    "recovery/turning metrics"
                ),
            },
        },
    }

    try:
        llm_input, input_meta = build_single_snapshot_input(source_record)
        rec["llm_input"] = llm_input
        rec["baseline_meta"]["input_meta"] = input_meta

        if skip_llm:
            rec["explanation"] = None
            rec["status"] = "ok"
        else:
            llm_task = _call_explanation_llm(llm_input, prompt_text, model)
            if timeout_sec > 0:
                rec["explanation"] = await asyncio.wait_for(
                    llm_task, timeout=timeout_sec
                )
            else:
                rec["explanation"] = await llm_task
            rec["status"] = "ok"
    except Exception:
        rec["status"] = "error"
        rec["error"] = traceback.format_exc()
        rec["explanation"] = None

    rec["elapsed_sec"] = round(time.time() - t0, 2)
    return rec


async def _run_generation(records: List[Dict[str, Any]], existing: List[Dict[str, Any]],
                          output: str, prompt_text: str, prompt_sha256: str,
                          model: str, workers: int,
                          skip_llm: bool, timeout_sec: float) -> Tuple[int, int]:
    input_order = [r.get("query_id") for r in records if r.get("query_id")]
    records_by_id: Dict[str, Dict[str, Any]] = {
        r["query_id"]: r for r in existing if isinstance(r, dict) and r.get("query_id")
    }

    semaphore = asyncio.Semaphore(max(1, workers))

    async def _bounded(source: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            return await _build_record(
                source, prompt_text, prompt_sha256, model, skip_llm, timeout_sec,
            )

    tasks = [asyncio.create_task(_bounded(r)) for r in records]
    n_ok = n_err = 0
    total = len(tasks)

    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        rec = await task
        qid = rec.get("query_id")
        if qid:
            records_by_id[qid] = rec
        ok = rec.get("status") == "ok"
        n_ok += int(ok)
        n_err += int(not ok)
        mark = "OK " if ok else "ERR"
        print(f"[{i}/{total}] {mark} {qid} elapsed={rec.get('elapsed_sec')}s")
        _flush(output, records_by_id, input_order)

    return n_ok, n_err


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="Source full NS evidence JSON")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Output B4 single-snapshot evidence/explanation JSON")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help=f"OpenAI model for B4 explanations (default: {DEFAULT_MODEL})")
    p.add_argument("--query-ids", nargs="+",
                   help="Run only these query_ids")
    p.add_argument("--limit", type=int,
                   help="Cap number of source records after filtering")
    p.add_argument("--workers", type=int, default=4,
                   help="Concurrent explanation LLM calls")
    p.add_argument("--skip-llm", action="store_true",
                   help="Write current-only prompts/evidence without LLM calls")
    p.add_argument("--timeout-sec", type=float, default=180.0,
                   help="Per-record LLM timeout in seconds; <=0 disables")
    p.add_argument("--resume", action="store_true",
                   help="Skip query_ids already present in --output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _load_dotenv()

    records = _filter_records(_load_records(args.input), args.query_ids, args.limit)
    if not records:
        print("No records match filters; nothing to do.")
        return 0

    prompt_text, prompt_sha256 = _embedded_prompt()

    existing: List[Dict[str, Any]] = []
    if args.resume and os.path.exists(args.output):
        existing = _load_existing(args.output)
        done = {r.get("query_id") for r in existing if isinstance(r, dict)}
        records = [r for r in records if r.get("query_id") not in done]
        print(f"[resume] {len(done)} already done; {len(records)} remaining")

    if not records:
        print("Nothing to do.")
        return 0

    print(
        "Generating B4/A1 single-snapshot records "
        f"from {args.input} -> {args.output}"
    )
    print(
        f"records={len(records)} workers={args.workers} "
        f"skip_llm={args.skip_llm} model={args.model}"
    )
    print(f"prompt_source=embedded:SINGLE_SNAPSHOT_PROMPT")

    t0 = time.time()
    n_ok, n_err = asyncio.run(
        _run_generation(
            records, existing, args.output, prompt_text, prompt_sha256,
            args.model, args.workers, args.skip_llm, args.timeout_sec,
        )
    )
    dt = time.time() - t0
    print(f"\nDone. ok={n_ok} err={n_err} elapsed={dt:.1f}s -> {args.output}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
