#!/usr/bin/env python3
"""
Generate A2/A3 channel-ablation explanation records.

A2 = PCTL only:
  - LLM input contains Query + PCTL Analysis Results only.
  - Algorithm State and Derived Metrics Summary are absent.
  - Evidence available to the verifier contains only the PCTL metrics that were
    actually shown to the model.

A3 = Derived metrics only:
  - LLM input contains Query + Algorithm State + Derived Metrics Summary.
  - PCTL Analysis Results are absent.
  - Evidence available to the verifier contains only the derived metrics that
    were actually shown to the model.

Both variants use embedded prompt strings derived from ns_prompt.md and build
their LLM input from structured evidence, rather than passing full inputs and
asking the model to ignore unavailable channels.
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
EVAL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(EVAL_DIR)
sys.path.insert(0, REPO_DIR)

DEFAULT_INPUT = os.path.join(EVAL_DIR, "faithfulness", "evidence_240.json")
DEFAULT_MODEL = "gpt-5.4-mini"

ABLATION_DEFAULT_OUTPUTS = {
    "a2": os.path.join(SCRIPT_DIR, "a2_pctl_only", "evidence_240.json"),
    "a3": os.path.join(SCRIPT_DIR, "a3_derived_only", "evidence_240.json"),
}

PCTL_SECTION_HEADER = "PCTL Analysis Results:"
DERIVED_SECTION_HEADER = "Derived Metrics Summary:"
TURNING_INTERVAL = "DERIVED: TURNING_INTERVAL"

A2_PCTL_ONLY_PROMPT = """You are generating an operations-facing dispatch explanation for a paratransit service management expert.

Audience:
- The reader has no background in computer science or formal methods.
- Write like an operational dispatch justification, not a technical explanation.
- Keep the explanation concise and practical, like a dispatcher explaining a decision.

Inputs:
- Query
- PCTL Analysis Results

A2 PCTL-only ablation setting:
- The input contains only formal probability/risk evidence for current and reference snapshots.
- Algorithm State is intentionally unavailable. Do not mention planner confidence, environment-change status, or algorithm phase.
- Derived Metrics Summary is intentionally unavailable. Do not cite or infer ETA, deadhead time, route burden, dropoff slack, workload, traffic level, service-path disruption, or recovery interval unless explicitly present in the PCTL labels.
- If the query asks for information that requires unavailable derived metrics, say that the PCTL-only evidence cannot determine that detail, then summarize the relevant probability/risk evidence if any exists.

Main rules:
- Answer the actual query directly.
- Shape the explanation around the asked query.
- Focus on the 1 to 2 most important probability/risk reasons only.
- Keep the explanation tight and avoid repeating similar points.
- If the query contains an incorrect or inconsistent assumption about the assignment, explicitly point out the mistake and clarify what the PCTL evidence can and cannot show.

Structure guidance:
- Use up to three paragraphs.
- Give each paragraph a short bold heading that fits the content.
- Include an alternative-vehicle paragraph only if the query asks for a comparison or the PCTL evidence makes that contrast necessary.
- Return the final explanation in GitHub-flavored Markdown.
- When current and reference probability values are available, explain the temporal change using those probability/risk values.
- Be concise, natural, and operations-facing.

Operational language:
- Do not use MDP_t, MDP_{t-n}, non-stationary, cross-snapshot, snapshot, PCTL, formulas, BNN, or other system-internal method names.
- Do not say "the model thinks," "the AI thinks," or "the planner's understanding."
- It is acceptable to say "the system currently estimates" or "at the reference update, the system estimated" when referring to risk/probability values.
- Use plain language such as chance of delay, risk of delay, service completion chance, and downstream risk.

Probability rules:
- Convert probabilities to percentages with at most one decimal place.
- Prefer pickup-delay or dropoff-delay risk when relevant.
- If a probability is at or near saturation (e.g., >=99% or <=1%), express it qualitatively (e.g., "near certainty" or "almost no risk") instead of exact percentages; when multiple options fall in this range, emphasize relative differences rather than treating them as identical.
- Tie probability statements to operational meaning whenever possible.
- When a probability reflects a future event, explain it as downstream operational risk after taking this request.

Repetition control:
- If an "Already shown to the user in this request" section is present, do not unnecessarily repeat detailed metrics or full argument structure.
- Focus on what the PCTL-only evidence can support."""

A3_DERIVED_ONLY_PROMPT = """You are generating an operations-facing dispatch explanation for a paratransit service management expert.

Audience:
- The reader has no background in computer science or formal methods.
- Write like an operational dispatch justification, not a technical explanation.
- Keep the explanation concise and practical, like a dispatcher explaining a decision.

Inputs:
- Query
- Algorithm State
- Derived Metrics Summary

A3 derived-metrics-only ablation setting:
- The input contains operational derived metrics and algorithm state, but no PCTL probability/risk analysis.
- Do not cite pickup-delay probability, dropoff-delay probability, service-completion probability, or any other formal risk probability unless it appears directly in the derived metrics.
- Use Algorithm State as part of the derived-metrics channel: environment-change status, algorithm phase, and planner confidence may be used when present.
- If the query asks for a probability/risk comparison that requires PCTL Analysis Results, say that the derived-only evidence cannot determine that probability, then summarize the relevant operational metrics if any exist.

Main rules:
- Answer the actual query directly.
- Shape the explanation around the asked query.
- Focus on the 1 to 2 most important operational reasons only.
- Keep the explanation tight and avoid repeating similar points.
- If the query contains an incorrect or inconsistent assumption about the assignment, explicitly point out the mistake and clarify the correct situation before answering.

Structure guidance:
- Use up to three paragraphs.
- Give each paragraph a short bold heading that fits the content.
- Include an alternative-vehicle paragraph only if the query asks for that comparison or the derived evidence makes that contrast necessary.
- Return the final explanation in GitHub-flavored Markdown.
- When current and reference derived values are available, explain change over time using concrete operational comparisons.
- Be concise, natural, and operations-facing.
- Anyone with no formal-methods background should be able to understand every sentence.

Operational language:
- Do not use MDP_t, MDP_{t-n}, non-stationary, cross-snapshot, snapshot, PCTL, formulas, BNN, or other system-internal method names.
- Do not say "the model thinks," "the AI thinks," or "the planner's understanding."
- It is acceptable to say "the system currently estimates" or "at the reference update, the system estimated" when referring to ETA, travel time, or operational values.
- Use plain operational language such as:
  - the environment has changed / the environment has not changed
  - stable phase / adapting after the change / stable again after the change
  - expected passenger wait
  - expected trip completion
  - can begin serving sooner
  - schedule margin

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
- A turning/recovery interval should be stated as a request interval when present.

Repetition control:
- If an "Already shown to the user in this request" section is present, do not unnecessarily repeat detailed metrics or full argument structure.
- Focus on what the derived-only evidence can support."""

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
        TURNING_INTERVAL,
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


def _read_records(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        records = json.load(f)
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON array")
    return records


def _filter_records(records: List[Dict[str, Any]], query_ids: Optional[List[str]],
                    limit: Optional[int]) -> List[Dict[str, Any]]:
    out = records
    if query_ids:
        qids = set(query_ids)
        out = [r for r in out if r.get("query_id") in qids]
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


def _prompt_for_ablation(ablation: str) -> Tuple[str, str, str]:
    if ablation == "a2":
        text = A2_PCTL_ONLY_PROMPT.strip()
        name = "A2 PCTL only"
    elif ablation == "a3":
        text = A3_DERIVED_ONLY_PROMPT.strip()
        name = "A3 derived metrics only"
    else:
        raise ValueError(f"unknown ablation {ablation!r}")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, digest, name


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


def _extract_visible_pctl_specs(record: Dict[str, Any]
                                ) -> List[Tuple[str, Optional[List[str]]]]:
    evidence = record.get("evidence", {}).get("E_t", {}) or {}
    lines = _section_lines(
        record.get("llm_input") or "",
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
        if current_metric and "MDP_t (current model):" in line:
            specs.append((current_metric, _vehicle_keys_in_current_side(line)))
            current_metric = None
        elif current_metric and "Current model:" in line:
            specs.append((current_metric, _vehicle_keys_in_current_side(line)))
            current_metric = None
    if specs:
        return specs
    return [
        (f, None) for f in (record.get("pctl_formulas") or [])
        if f in evidence and not str(f).startswith("DERIVED:")
    ]


def _extract_visible_derived_specs(record: Dict[str, Any]
                                   ) -> List[Tuple[str, Optional[List[str]]]]:
    evidence = record.get("evidence", {}).get("E_t", {}) or {}
    lines = _section_lines(record.get("llm_input") or "", DERIVED_SECTION_HEADER)
    specs: List[Tuple[str, Optional[List[str]]]] = []
    for line in lines:
        m = re.match(r"^\s{4}([A-Z0-9_]+)(?:\s+\([^)]+\))?:", line)
        if not m:
            continue
        metric = f"DERIVED: {m.group(1)}"
        if metric in evidence:
            specs.append((metric, _vehicle_keys_in_current_side(line)))
    if specs:
        return specs
    return [
        (f, None) for f in (record.get("pctl_formulas") or [])
        if f in evidence and str(f).startswith("DERIVED:")
    ]


def _extract_algorithm_state(record: Dict[str, Any]) -> List[str]:
    llm_input = record.get("llm_input") or ""
    lines = []
    in_block = False
    for line in llm_input.splitlines():
        stripped = line.strip()
        if stripped == "Algorithm State:":
            in_block = True
            continue
        if in_block and stripped in (PCTL_SECTION_HEADER, DERIVED_SECTION_HEADER):
            break
        if in_block and stripped:
            lines.append(line)
    return lines


def _derived_display_name(metric: str) -> str:
    return metric.replace("DERIVED: ", "")


def _format_turning_interval(raw: Any) -> str:
    if isinstance(raw, dict):
        interval = raw.get("display_interval")
        if isinstance(interval, list) and len(interval) >= 2:
            return f"request {interval[0]}-{interval[1]}"
        rec = raw.get("recovery_request_interval")
        if isinstance(rec, list) and len(rec) >= 2:
            return f"request {rec[0]}-{rec[1]}"
    return _format_metric_values(raw)


def _format_pctl_section(record: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    evidence = record.get("evidence", {}) or {}
    e_t = evidence.get("E_t", {}) or {}
    e_prev = evidence.get("E_t_prev", {}) or {}
    specs = _extract_visible_pctl_specs(record)
    lines = [PCTL_SECTION_HEADER]
    rendered = []
    if not specs:
        lines.append("No PCTL metrics available.")
        return lines, rendered
    for metric, keys in specs:
        if metric not in e_t:
            continue
        lines.append(f"  {metric}:")
        lines.append(
            "    Current model: "
            f"{_format_metric_values(e_t[metric], pctl=True, keys=keys)}"
        )
        if metric in e_prev:
            lines.append(
                "    Reference model: "
                f"{_format_metric_values(e_prev[metric], pctl=True, keys=keys)}"
            )
        rendered.append(metric)
    if not rendered:
        lines.append("No PCTL metrics available.")
    return lines, rendered


def _format_derived_section(record: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    evidence = record.get("evidence", {}) or {}
    e_t = evidence.get("E_t", {}) or {}
    e_prev = evidence.get("E_t_prev", {}) or {}
    spec_map = {m: keys for m, keys in _extract_visible_derived_specs(record)}
    lines = [DERIVED_SECTION_HEADER]
    rendered: List[str] = []
    for group_name, metrics in DERIVED_GROUPS:
        group_lines = []
        for metric in metrics:
            if metric not in spec_map or metric not in e_t:
                continue
            name = _derived_display_name(metric)
            if metric == TURNING_INTERVAL:
                group_lines.append(f"    {name}: {_format_turning_interval(e_t[metric])}")
                rendered.append(metric)
                continue
            unit = METRIC_UNITS.get(metric)
            label = f"{name} ({unit})" if unit else name
            current = _format_metric_values(e_t[metric], keys=spec_map[metric])
            if metric in e_prev:
                ref = _format_metric_values(e_prev[metric], keys=spec_map[metric])
                group_lines.append(f"    {label}: current={current} | reference={ref}")
            else:
                group_lines.append(f"    {label}: current={current}")
            rendered.append(metric)
        if group_lines:
            lines.append(f"  [{group_name}]")
            lines.extend(group_lines)
    if not rendered:
        lines.append("No derived metrics available.")
    return lines, rendered


def build_a2_input(record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    pctl_lines, pctl_metrics = _format_pctl_section(record)
    lines = [f"Query: {record.get('query_text') or ''}", "", *pctl_lines, ""]
    meta = {
        "input_builder": "a2_pctl_only",
        "included_pctl_metrics": pctl_metrics,
        "included_derived_metrics": [],
        "algorithm_state_fields": [],
        "excluded_sections": ["Algorithm State", "Derived Metrics Summary"],
    }
    return "\n".join(lines).strip() + "\n", meta


def build_a3_input(record: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    algo_lines = _extract_algorithm_state(record)
    derived_lines, derived_metrics = _format_derived_section(record)
    lines = [f"Query: {record.get('query_text') or ''}", "", "Algorithm State:"]
    if algo_lines:
        lines.extend(algo_lines)
    else:
        lines.append("  Algorithm state unavailable.")
    lines.extend(["", *derived_lines, ""])
    meta = {
        "input_builder": "a3_derived_only",
        "included_pctl_metrics": [],
        "included_derived_metrics": derived_metrics,
        "algorithm_state_fields": [line.strip().split(":", 1)[0] for line in algo_lines],
        "excluded_sections": ["PCTL Analysis Results"],
    }
    return "\n".join(lines).strip() + "\n", meta


def _build_ablation_evidence(record: Dict[str, Any], ablation: str) -> Dict[str, Any]:
    evidence = record.get("evidence") or {}
    e_t = evidence.get("E_t", {}) or {}
    e_prev = evidence.get("E_t_prev", {}) or {}
    if ablation == "a2":
        specs = _extract_visible_pctl_specs(record)
    elif ablation == "a3":
        specs = _extract_visible_derived_specs(record)
    else:
        raise ValueError(f"unknown ablation {ablation!r}")
    allowed = {metric: keys for metric, keys in specs}
    out_t = {}
    out_prev = {}
    for metric, keys in allowed.items():
        if metric in e_t:
            out_t[metric] = _filter_metric_value(e_t[metric], keys)
        if metric in e_prev:
            out_prev[metric] = _filter_metric_value(e_prev[metric], keys)
    return {
        "E_t": _jsonable(out_t),
        "E_t_prev": _jsonable(out_prev),
        "snapshot_meta": {
            "ablation": ablation,
            "source_has_E_t_prev": bool(e_prev),
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


async def _build_record(source_record: Dict[str, Any], args: argparse.Namespace,
                        prompt_text: str, prompt_sha256: str,
                        ablation_name: str) -> Dict[str, Any]:
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
        "pctl_formulas": list(_build_ablation_evidence(source_record, args.ablation)["E_t"].keys()),
        "evidence": _build_ablation_evidence(source_record, args.ablation),
        "ablation_meta": {
            "ablation_id": args.ablation.upper(),
            "name": ablation_name,
            "source_query_id": qid,
            "prompt_source": (
                "embedded:A2_PCTL_ONLY_PROMPT"
                if args.ablation == "a2" else "embedded:A3_DERIVED_ONLY_PROMPT"
            ),
            "prompt_sha256": prompt_sha256,
            "model": args.model,
            "input_group_policy": (
                {
                    "pctl_analysis_results": "current and reference snapshots",
                    "algorithm_state": "removed",
                    "derived_metrics_summary": "removed",
                }
                if args.ablation == "a2"
                else {
                    "pctl_analysis_results": "removed",
                    "algorithm_state": "kept",
                    "derived_metrics_summary": "current and reference snapshots",
                }
            ),
        },
    }
    try:
        if args.ablation == "a2":
            llm_input, input_meta = build_a2_input(source_record)
        else:
            llm_input, input_meta = build_a3_input(source_record)
        rec["llm_input"] = llm_input
        rec["ablation_meta"]["input_meta"] = input_meta

        if args.skip_llm:
            rec["explanation"] = None
            rec["status"] = "ok"
        else:
            task = _call_explanation_llm(llm_input, prompt_text, args.model)
            if args.timeout_sec > 0:
                rec["explanation"] = await asyncio.wait_for(task, args.timeout_sec)
            else:
                rec["explanation"] = await task
            rec["status"] = "ok"
    except Exception:
        rec["status"] = "error"
        rec["error"] = traceback.format_exc()
        rec["explanation"] = None
    rec["elapsed_sec"] = round(time.time() - t0, 2)
    return rec


async def _run(records: List[Dict[str, Any]], existing: List[Dict[str, Any]],
               args: argparse.Namespace, prompt_text: str,
               prompt_sha256: str, ablation_name: str) -> Tuple[int, int]:
    input_order = [r.get("query_id") for r in records if r.get("query_id")]
    records_by_id = {
        r["query_id"]: r for r in existing if isinstance(r, dict) and r.get("query_id")
    }
    semaphore = asyncio.Semaphore(max(1, args.workers))

    async def _bounded(source):
        async with semaphore:
            return await _build_record(source, args, prompt_text,
                                       prompt_sha256, ablation_name)

    tasks = [asyncio.create_task(_bounded(r)) for r in records]
    n_ok = n_err = 0
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        rec = await task
        qid = rec.get("query_id")
        if qid:
            records_by_id[qid] = rec
        ok = rec.get("status") == "ok"
        n_ok += int(ok)
        n_err += int(not ok)
        mark = "OK " if ok else "ERR"
        print(f"[{i}/{len(tasks)}] {mark} {qid} elapsed={rec.get('elapsed_sec')}s")
        _flush(args.output, records_by_id, input_order)
    return n_ok, n_err


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ablation", required=True, choices=["a2", "a3"],
                   help="A2=PCTL only; A3=derived metrics only")
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="Source full NS evidence JSON")
    p.add_argument("--output",
                   help="Output JSON path; defaults under evaluation/ablations")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--query-ids", nargs="+")
    p.add_argument("--limit", type=int)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-llm", action="store_true")
    p.add_argument("--timeout-sec", type=float, default=180.0)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _load_dotenv()
    if not args.output:
        args.output = ABLATION_DEFAULT_OUTPUTS[args.ablation]

    records = _filter_records(_read_records(args.input), args.query_ids, args.limit)
    if not records:
        print("No records match filters; nothing to do.")
        return 0

    existing: List[Dict[str, Any]] = []
    if args.resume and os.path.exists(args.output):
        existing = _load_existing(args.output)
        done = {r.get("query_id") for r in existing if isinstance(r, dict)}
        records = [r for r in records if r.get("query_id") not in done]
        print(f"[resume] {len(done)} already done; {len(records)} remaining")
    if not records:
        print("Nothing to do.")
        return 0

    prompt_text, prompt_sha256, ablation_name = _prompt_for_ablation(args.ablation)
    print(
        f"Generating {ablation_name} records from {args.input} -> {args.output}"
    )
    print(
        f"records={len(records)} workers={args.workers} "
        f"skip_llm={args.skip_llm} model={args.model}"
    )

    t0 = time.time()
    n_ok, n_err = asyncio.run(
        _run(records, existing, args, prompt_text, prompt_sha256, ablation_name)
    )
    dt = time.time() - t0
    print(f"\nDone. ok={n_ok} err={n_err} elapsed={dt:.1f}s -> {args.output}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
