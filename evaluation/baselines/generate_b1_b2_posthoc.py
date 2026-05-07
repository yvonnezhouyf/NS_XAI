#!/usr/bin/env python3
"""
Generate B1/B2 post-hoc baseline explanation records.

B1 = trajectory-only post-hoc:
  - LLM receives the executed assignment/trajectory state for the queried
    request only.
  - No PCTL probabilities, no cross-snapshot evidence, no planner policy stats.

B2 = trajectory + policy post-hoc:
  - LLM receives the same trajectory input as B1.
  - It also receives root planner policy/search statistics: visit count, visit
    share, and Q/value by vehicle.
  - Still no PCTL probabilities or cross-snapshot evidence.

Outputs intentionally match the faithfulness evidence schema so the existing
claim extractor, verifier, and Delta-Recovery scripts can be reused.
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
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(EVAL_DIR)
FAITHFULNESS_DIR = os.path.join(EVAL_DIR, "faithfulness")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, FAITHFULNESS_DIR)

from generate_evidence import SCENARIO_LOADERS, _to_jsonable  # noqa: E402

DEFAULT_INPUT = os.path.join(EVAL_DIR, "faithfulness", "evidence_240.json")
DEFAULT_MODEL = "gpt-5.4-mini"

DEFAULT_OUTPUTS = {
    "b1": os.path.join(SCRIPT_DIR, "b1_trajectory_only", "evidence_240.json"),
    "b2": os.path.join(SCRIPT_DIR, "b2_trajectory_policy", "evidence_240.json"),
}

B1_PROMPT = """You are generating an operations-facing dispatch explanation for a paratransit service management expert.

Baseline setting:
- You are a trajectory-only post-hoc baseline.
- You receive only the executed assignment and trajectory state at the queried request.
- You do not have PCTL probabilities, formal risk analysis, planner policy scores, or previous/reference-snapshot evidence.

Rules:
- Answer the query as directly as the trajectory data allows.
- If the query asks for delay risk, reliability, temporal change, adaptation, recovery, or a counterfactual reason that the trajectory alone cannot establish, say that this baseline cannot determine that from trajectory data alone.
- Then summarize the most relevant visible trajectory facts, such as chosen vehicle, closest vehicle, distance to pickup, pending requests, occupancy, request timing, and traffic level.
- Do not invent probabilities, risk percentages, before/after comparisons, or formal evidence.
- Use at most three short paragraphs with bold headings.
- Keep the language practical and dispatch-facing.
"""

B2_PROMPT = """You are generating an operations-facing dispatch explanation for a paratransit service management expert.

Baseline setting:
- You are a trajectory + policy post-hoc baseline.
- You receive the executed assignment/trajectory state and planner root search statistics for the queried request.
- The policy statistics include visit counts, visit shares, and Q/value scores by vehicle.
- You do not have PCTL probabilities, formal risk analysis, or previous/reference-snapshot evidence.

Rules:
- Answer the query as directly as the trajectory and policy statistics allow.
- Treat visit share and Q/value as planner search evidence, not as delay-risk probability.
- If the query asks for formal risk, temporal change, adaptation, recovery, or a before/after comparison, say that this baseline cannot determine that without cross-snapshot formal evidence.
- Then summarize the most relevant visible trajectory and policy facts.
- Do not invent probabilities, PCTL-style risk percentages, or previous/reference values.
- Use at most three short paragraphs with bold headings.
- Keep the language practical and dispatch-facing.
"""


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(REPO_DIR, ".env"))


def _prompt_for(baseline: str) -> Tuple[str, str, str]:
    if baseline == "b1":
        text = B1_PROMPT.strip()
        name = "B1 trajectory-only post-hoc"
    elif baseline == "b2":
        text = B2_PROMPT.strip()
        name = "B2 trajectory + policy post-hoc"
    else:
        raise ValueError(f"unknown baseline {baseline!r}")
    return text, hashlib.sha256(text.encode("utf-8")).hexdigest(), name


def _read_records(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON array")
    return data


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


def _fmt_num(value: Any, digits: int = 2) -> str:
    if isinstance(value, (int, float)):
        if abs(float(value) - round(float(value))) < 1e-9:
            return str(int(round(float(value))))
        return f"{float(value):.{digits}f}"
    return "unavailable" if value is None else str(value)


def _vehicle_sort_key(item: Tuple[str, Any]) -> Tuple[int, str]:
    key, _ = item
    if isinstance(key, str) and re.fullmatch(r"V\d+", key):
        return (int(key[1:]), key)
    return (999, str(key))


def _format_vehicle_map(values: Dict[str, Any], digits: int = 2) -> str:
    items = sorted(values.items(), key=_vehicle_sort_key)
    return "{" + ", ".join(f"{k}={_fmt_num(v, digits)}" for k, v in items) + "}"


def _assignment_detail(mcts_data: Dict[str, Any], epoch: int) -> Dict[str, Any]:
    details = mcts_data.get("env_data", {}).get("assignment_details", [])
    if isinstance(details, list):
        if 0 <= epoch < len(details):
            return details[epoch] or {}
        for d in details:
            if isinstance(d, dict) and d.get("decision_epoch") == epoch:
                return d
    elif isinstance(details, dict):
        return details.get(epoch) or details.get(str(epoch)) or {}
    return {}


def _tree_for_epoch(mcts_data: Dict[str, Any], epoch: int) -> Optional[Dict[str, Any]]:
    trees = mcts_data.get("per_step_trees", {}) or {}
    for key, tree in trees.items():
        if isinstance(key, str) and "|" in key:
            try:
                key_epoch = int(key.rsplit("|", 1)[1])
            except ValueError:
                continue
            if key_epoch == epoch and isinstance(tree, dict):
                return tree
        elif key == epoch and isinstance(tree, dict):
            return tree
    return None


def _vehicle_option_maps(detail: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    options = detail.get("vehicle_options") or []
    maps = {
        "distance": {},
        "pending": {},
        "occupancy": {},
        "completed": {},
        "capacity": {},
    }
    for opt in options:
        if not isinstance(opt, dict):
            continue
        vid = opt.get("vehicle_id")
        if vid is None:
            continue
        key = f"V{vid}"
        maps["distance"][key] = opt.get("distance")
        maps["pending"][key] = opt.get("pending_requests")
        maps["occupancy"][key] = opt.get("current_occupancy")
        maps["completed"][key] = opt.get("completed_requests")
        maps["capacity"][key] = opt.get("capacity")
    return maps


def _policy_maps(tree: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    visits: Dict[str, Any] = {}
    shares: Dict[str, Any] = {}
    q_values: Dict[str, Any] = {}
    values: Dict[str, Any] = {}
    if not isinstance(tree, dict):
        return {
            "visits": visits,
            "visit_share": shares,
            "q_value": q_values,
            "value": values,
        }
    children = [c for c in (tree.get("children") or []) if isinstance(c, dict)]
    total_visits = sum(float(c.get("visits") or 0.0) for c in children)
    for child in children:
        action = child.get("action")
        if action is None:
            continue
        key = f"V{action}"
        v = float(child.get("visits") or 0.0)
        visits[key] = v
        shares[key] = (v / total_visits) if total_visits > 0 else 0.0
        q_values[key] = child.get("q_value")
        values[key] = child.get("value")
    return {
        "visits": visits,
        "visit_share": shares,
        "q_value": q_values,
        "value": values,
    }


def _build_baseline_evidence(detail: Dict[str, Any],
                             policy: Optional[Dict[str, Dict[str, Any]]],
                             baseline: str) -> Dict[str, Any]:
    maps = _vehicle_option_maps(detail)
    e_t: Dict[str, Any] = {
        "BASELINE: ASSIGNED_VEHICLE": (
            f"V{detail.get('assigned_vehicle')}"
            if detail.get("assigned_vehicle") is not None else None
        ),
        "BASELINE: CLOSEST_VEHICLE": (
            f"V{detail.get('closest_vehicle')}"
            if detail.get("closest_vehicle") is not None else None
        ),
        "BASELINE: DISTANCE_TO_PICKUP_BY_VEHICLE": maps["distance"],
        "BASELINE: PENDING_REQUESTS_BY_VEHICLE": maps["pending"],
        "BASELINE: CURRENT_OCCUPANCY_BY_VEHICLE": maps["occupancy"],
        "BASELINE: COMPLETED_REQUESTS_BY_VEHICLE": maps["completed"],
        "BASELINE: VEHICLE_CAPACITY_BY_VEHICLE": maps["capacity"],
        "BASELINE: REQUEST_TIME": detail.get("request_time"),
        "BASELINE: EARLIEST_PICKUP": detail.get("earliest_pickup"),
        "BASELINE: LATEST_DROPOFF": detail.get("latest_dropoff"),
        "BASELINE: TRAFFIC_LEVEL": detail.get("traffic_level"),
    }
    if baseline == "b2" and policy:
        e_t.update({
            "BASELINE: POLICY_VISITS_BY_VEHICLE": policy.get("visits", {}),
            "BASELINE: POLICY_VISIT_SHARE_BY_VEHICLE": policy.get("visit_share", {}),
            "BASELINE: POLICY_Q_VALUE_BY_VEHICLE": policy.get("q_value", {}),
            "BASELINE: POLICY_TOTAL_VALUE_BY_VEHICLE": policy.get("value", {}),
        })
    return {
        "E_t": _to_jsonable(e_t),
        "E_t_prev": {},
        "snapshot_meta": {
            "baseline": baseline,
            "available_evidence_scope": (
                "executed trajectory only"
                if baseline == "b1"
                else "executed trajectory plus planner root policy statistics"
            ),
            "source_has_cross_snapshot_evidence": False,
            "source_has_pctl_evidence": False,
        },
    }


def _build_b1_input(record: Dict[str, Any], detail: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    maps = _vehicle_option_maps(detail)
    assigned = detail.get("assigned_vehicle")
    closest = detail.get("closest_vehicle")
    lines = [
        f"Query: {record.get('query_text') or ''}",
        "",
        "Baseline Input: Executed Trajectory Only",
        f"Decision epoch: {record.get('epoch')}",
        f"Executed assignment: chosen=V{assigned}; closest=V{closest}",
        (
            "Request timing: "
            f"request_time={_fmt_num(detail.get('request_time'))}; "
            f"earliest_pickup={_fmt_num(detail.get('earliest_pickup'))}; "
            f"latest_dropoff={_fmt_num(detail.get('latest_dropoff'))}"
        ),
        f"Traffic level: {_fmt_num(detail.get('traffic_level'), digits=2)}",
        "",
        "Vehicle trajectory state before assignment:",
        f"  DISTANCE_TO_PICKUP_BY_VEHICLE (minutes): {_format_vehicle_map(maps['distance'])}",
        f"  PENDING_REQUESTS_BY_VEHICLE: {_format_vehicle_map(maps['pending'], digits=0)}",
        f"  CURRENT_OCCUPANCY_BY_VEHICLE: {_format_vehicle_map(maps['occupancy'], digits=0)}",
        f"  COMPLETED_REQUESTS_BY_VEHICLE: {_format_vehicle_map(maps['completed'], digits=0)}",
        "",
        "Unavailable to this baseline: PCTL probabilities, formal delay-risk analysis, reference-snapshot values, temporal deltas, recovery intervals, and planner policy scores.",
        "",
    ]
    meta = {
        "input_builder": "b1_trajectory_only",
        "included_sections": ["Executed Trajectory Only"],
        "excluded_sections": [
            "PCTL Analysis Results",
            "Derived Metrics Summary",
            "Reference snapshot",
            "Planner policy statistics",
        ],
    }
    return "\n".join(lines), meta


def _build_b2_input(record: Dict[str, Any], detail: Dict[str, Any],
                    policy: Dict[str, Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    base, meta = _build_b1_input(record, detail)
    policy_lines = [
        "Planner policy/search statistics at this request:",
        f"  POLICY_VISITS_BY_VEHICLE: {_format_vehicle_map(policy.get('visits', {}), digits=0)}",
        f"  POLICY_VISIT_SHARE_BY_VEHICLE: {_format_vehicle_map(policy.get('visit_share', {}), digits=4)}",
        f"  POLICY_Q_VALUE_BY_VEHICLE: {_format_vehicle_map(policy.get('q_value', {}), digits=4)}",
        "",
        "Policy note: visit share is planner search concentration, not delay-risk probability.",
        "",
    ]
    base = base.replace(
        "Unavailable to this baseline: PCTL probabilities, formal delay-risk analysis, reference-snapshot values, temporal deltas, recovery intervals, and planner policy scores.",
        "Unavailable to this baseline: PCTL probabilities, formal delay-risk analysis, reference-snapshot values, temporal deltas, and recovery intervals.",
    )
    text = base.rstrip() + "\n\n" + "\n".join(policy_lines)
    meta["input_builder"] = "b2_trajectory_policy"
    meta["included_sections"] = [
        "Executed Trajectory Only",
        "Planner policy/search statistics",
    ]
    meta["excluded_sections"] = [
        "PCTL Analysis Results",
        "Derived Metrics Summary",
        "Reference snapshot",
    ]
    return text, meta


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


async def _build_record(source_record: Dict[str, Any], mcts_data_by_scenario: Dict[int, Dict[str, Any]],
                        args: argparse.Namespace, prompt_text: str,
                        prompt_sha256: str, baseline_name: str) -> Dict[str, Any]:
    t0 = time.time()
    qid = source_record.get("query_id")
    scenario = int(source_record.get("scenario"))
    epoch = int(source_record.get("epoch"))
    mcts_data = mcts_data_by_scenario[scenario]
    detail = _assignment_detail(mcts_data, epoch)
    policy = _policy_maps(_tree_for_epoch(mcts_data, epoch))
    evidence = _build_baseline_evidence(
        detail, policy if args.baseline == "b2" else None, args.baseline
    )

    rec: Dict[str, Any] = {
        "query_id": qid,
        "type_k": source_record.get("type_k"),
        "scenario": scenario,
        "query_text": source_record.get("query_text"),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "epoch": epoch,
        "classification_label": source_record.get("classification_label"),
        "resolved_vehicle_targets": source_record.get("resolved_vehicle_targets", []),
        "pctl_formulas": list(evidence["E_t"].keys()),
        "evidence": evidence,
        "baseline_meta": {
            "baseline_id": args.baseline.upper(),
            "name": baseline_name,
            "source_query_id": qid,
            "prompt_source": (
                "embedded:B1_PROMPT" if args.baseline == "b1"
                else "embedded:B2_PROMPT"
            ),
            "prompt_sha256": prompt_sha256,
            "model": args.model,
            "input_group_policy": (
                {
                    "executed_trajectory": "kept",
                    "planner_policy_statistics": "removed",
                    "pctl_analysis_results": "removed",
                    "cross_snapshot_reference": "removed",
                }
                if args.baseline == "b1"
                else {
                    "executed_trajectory": "kept",
                    "planner_policy_statistics": "kept",
                    "pctl_analysis_results": "removed",
                    "cross_snapshot_reference": "removed",
                }
            ),
        },
    }

    try:
        if args.baseline == "b1":
            llm_input, input_meta = _build_b1_input(source_record, detail)
        else:
            llm_input, input_meta = _build_b2_input(source_record, detail, policy)
        rec["llm_input"] = llm_input
        rec["baseline_meta"]["input_meta"] = input_meta

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
               mcts_data_by_scenario: Dict[int, Dict[str, Any]],
               args: argparse.Namespace, prompt_text: str,
               prompt_sha256: str, baseline_name: str) -> Tuple[int, int]:
    input_order = [r.get("query_id") for r in records if r.get("query_id")]
    records_by_id = {
        r["query_id"]: r for r in existing
        if isinstance(r, dict) and r.get("query_id")
    }
    semaphore = asyncio.Semaphore(max(1, args.workers))

    async def _bounded(source):
        async with semaphore:
            return await _build_record(
                source, mcts_data_by_scenario, args,
                prompt_text, prompt_sha256, baseline_name
            )

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
    p.add_argument("--baseline", required=True, choices=["b1", "b2"])
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="Source full NS evidence JSON for query metadata")
    p.add_argument("--output",
                   help="Output JSON path; defaults under evaluation/baselines")
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
        args.output = DEFAULT_OUTPUTS[args.baseline]

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

    print("Loading scenario data...")
    import main_paratransit_demo as demo
    needed_scenarios = sorted({int(r.get("scenario")) for r in records})
    mcts_data_by_scenario: Dict[int, Dict[str, Any]] = {}
    for scenario in needed_scenarios:
        loader_name = SCENARIO_LOADERS.get(scenario)
        if loader_name is None:
            raise ValueError(f"no loader for scenario {scenario}")
        print(f"  scenario {scenario}: {loader_name}")
        mcts_data_by_scenario[scenario] = getattr(demo, loader_name)(use_saved=True)

    prompt_text, prompt_sha256, baseline_name = _prompt_for(args.baseline)
    print(
        f"Generating {baseline_name} records from {args.input} -> {args.output}"
    )
    print(
        f"records={len(records)} workers={args.workers} "
        f"skip_llm={args.skip_llm} model={args.model}"
    )
    t0 = time.time()
    n_ok, n_err = asyncio.run(
        _run(records, existing, mcts_data_by_scenario, args,
             prompt_text, prompt_sha256, baseline_name)
    )
    dt = time.time() - t0
    print(f"\nDone. ok={n_ok} err={n_err} elapsed={dt:.1f}s -> {args.output}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
