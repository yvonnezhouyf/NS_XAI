#!/usr/bin/env python3
"""
Paraphrase robustness generation and scoring.

Generation:
  For every core query record, generate or load a 5-query paraphrase set. Keep
  the method's evidence fixed and replace only the Query line in the LLM input.
  This isolates explanation robustness to wording changes.

Scoring:
  After running claim_extractor.py on the generated paraphrase evidence, compute
  pairwise robustness within each base_id group:
    - numerical-value Jaccard: set of numerical values cited, with tolerance
    - claim-set Jaccard: extracted atomic claim units, with numeric tolerance

Example:
  python evaluation/robustness/paraphrase_robustness.py generate \
    --method ours --workers 4 --resume

  python evaluation/faithfulness/claim_extractor.py \
    --input evaluation/robustness/paraphrase/ours_evidence.json \
    --output evaluation/robustness/paraphrase/ours_claims.json \
    --workers 8 --resume

  python evaluation/robustness/paraphrase_robustness.py score \
    --evidence evaluation/robustness/paraphrase/ours_evidence.json \
    --claims evaluation/robustness/paraphrase/ours_claims.json \
    --output evaluation/robustness/paraphrase/ours_summary.json
"""

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from itertools import combinations
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(EVAL_DIR)
BASELINES_DIR = os.path.join(EVAL_DIR, "baselines")
B4_DIR = os.path.join(BASELINES_DIR, "b4_single_snapshot")
sys.path.insert(0, REPO_DIR)
sys.path.insert(0, BASELINES_DIR)
sys.path.insert(0, B4_DIR)

DEFAULT_LABELS = os.path.join(EVAL_DIR, "labels.txt")
DEFAULT_QUERIES = os.path.join(EVAL_DIR, "queries.txt")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "paraphrase")
DEFAULT_PARAPHRASE_CACHE = os.path.join(DEFAULT_OUTPUT_DIR, "generated_paraphrases.json")
DEFAULT_MODEL = "gpt-5.4-mini"

DEFAULT_INPUTS = {
    "ours": os.path.join(EVAL_DIR, "faithfulness", "evidence_240.json"),
    "b1": os.path.join(BASELINES_DIR, "b1_trajectory_only", "evidence_240.json"),
    "b2": os.path.join(BASELINES_DIR, "b2_trajectory_policy", "evidence_240.json"),
    "b4": os.path.join(BASELINES_DIR, "b4_single_snapshot", "evidence_240.json"),
}

METHOD_NAMES = {
    "ours": "Full NS-XAI",
    "b1": "B1 trajectory-only post-hoc",
    "b2": "B2 trajectory + policy post-hoc",
    "b4": "B4/A1 single-snapshot",
}

PARAPHRASE_PROMPT = """Generate four paraphrases of the dispatch query.

Requirements:
- Preserve the exact meaning.
- Preserve all vehicle IDs, request numbers, epoch numbers, scenario cues, and comparison direction.
- Do not introduce new facts.
- Do not answer the query.
- Keep each paraphrase natural and concise.

Return JSON only:
{"paraphrases": ["...", "...", "...", "..."]}

QUERY:
{query}
"""


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(os.path.join(REPO_DIR, ".env"))


def _read_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def _write_json_atomic(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _read_labels(path: str) -> Dict[str, Dict[str, str]]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}


def _read_queries(path: str) -> Dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r["query"] for r in csv.DictReader(f, delimiter="\t")}


def _paraphrase_groups(labels: Dict[str, Dict[str, str]],
                       queries: Dict[str, str]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for qid, row in labels.items():
        if qid not in queries:
            continue
        grouped[row["base_id"]].append({
            "query_id": qid,
            "query_text": queries[qid],
            "base_id": row["base_id"],
            "paraphrase_id": int(row["paraphrase_id"]),
            "type_k": int(row["type_k"]),
            "scenario_label": row.get("scenario"),
            "is_adversarial": int(row.get("is_adversarial") or 0),
        })
    return {
        base_id: sorted(items, key=lambda x: x["paraphrase_id"])
        for base_id, items in grouped.items()
    }


def _load_paraphrase_cache(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"items": {}}
    data = _read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    data.setdefault("items", {})
    return data


def _cached_paraphrases_are_usable(cached: Dict[str, Any], source_query_text: str,
                                   skip_llm: bool) -> bool:
    paraphrases = cached.get("paraphrases")
    if not isinstance(paraphrases, list) or len(paraphrases) < 5:
        return False
    if cached.get("source_query_text") != source_query_text:
        return False
    if skip_llm:
        return True
    sources = {p.get("source") for p in paraphrases if isinstance(p, dict)}
    return "deterministic_fallback" not in sources


def _flush_paraphrase_cache(path: str, cache: Dict[str, Any]) -> None:
    cache["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _write_json_atomic(path, cache)


def _load_existing(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    data = _read_json(path)
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


def _replace_query_line(llm_input: str, query_text: str) -> str:
    text = llm_input or ""
    replacement = f"Query: {query_text}"
    if re.search(r"^Query:.*$", text, flags=re.M):
        return re.sub(r"^Query:.*$", replacement, text, count=1, flags=re.M)
    return replacement + "\n\n" + text


def _fallback_paraphrases(source_query_id: str, query_text: str) -> List[Dict[str, Any]]:
    """Deterministic fallback used for --skip-llm smoke tests."""
    templates = [
        query_text,
        f"Can you explain this dispatch decision: {query_text}",
        f"In operational terms, {query_text[0].lower() + query_text[1:] if query_text else query_text}",
        f"What is the reason behind this: {query_text}",
        f"Could you clarify the dispatch rationale here: {query_text}",
    ]
    return [
        {
            "query_id": f"{source_query_id}_p{i}",
            "query_text": text,
            "base_id": source_query_id,
            "paraphrase_id": i,
            "source": "deterministic_fallback",
        }
        for i, text in enumerate(templates)
    ]


async def _call_paraphrase_llm(query_text: str, model: str) -> List[str]:
    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(query=query_text)}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(response.choices[0].message.content or "{}")
    paras = data.get("paraphrases")
    if not isinstance(paras, list):
        raise ValueError("paraphrase response missing list field 'paraphrases'")
    cleaned = []
    for item in paras:
        if isinstance(item, str) and item.strip():
            cleaned.append(item.strip())
    if len(cleaned) < 4:
        raise ValueError(f"expected 4 paraphrases, got {len(cleaned)}")
    return cleaned[:4]


async def _call_paraphrase_llm_with_retry(query_text: str, model: str,
                                          max_retries: int) -> List[str]:
    delay = 1.0
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return await _call_paraphrase_llm(query_text, model)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(delay + random.uniform(0.0, 0.5))
            delay *= 2
    assert last_exc is not None
    raise last_exc


async def _generated_paraphrases_for_source(
    source: Dict[str, Any],
    args: argparse.Namespace,
    cache: Dict[str, Any],
) -> List[Dict[str, Any]]:
    source_qid = source["query_id"]
    original = source.get("query_text") or ""
    cached = cache.get("items", {}).get(source_qid)
    if (
        isinstance(cached, dict)
        and _cached_paraphrases_are_usable(cached, original, args.skip_llm)
    ):
        return cached["paraphrases"]

    if args.skip_llm:
        paraphrases = _fallback_paraphrases(source_qid, original)
    else:
        generated = await _call_paraphrase_llm_with_retry(
            original, args.model, args.max_retries
        )
        paraphrases = [
            {
                "query_id": f"{source_qid}_p0",
                "query_text": original,
                "base_id": source_qid,
                "paraphrase_id": 0,
                "source": "original",
            }
        ]
        for i, text in enumerate(generated, 1):
            paraphrases.append({
                "query_id": f"{source_qid}_p{i}",
                "query_text": text,
                "base_id": source_qid,
                "paraphrase_id": i,
                "source": "llm_generated",
            })

    cache.setdefault("items", {})[source_qid] = {
        "source_query_id": source_qid,
        "source_query_text": original,
        "model": args.model,
        "paraphrases": paraphrases,
    }
    _flush_paraphrase_cache(args.paraphrase_cache, cache)
    return paraphrases


def _prompt_signature(method: str) -> Tuple[str, str, str]:
    if method == "ours":
        from use_cases.paratransit.config import ParatransitConfig
        prompt_id = ParatransitConfig.EXPLANATION_GENERATION_PROMPT_ID
        return "stored_prompt", prompt_id, f"openai:{prompt_id}"
    if method in ("b1", "b2"):
        from generate_b1_b2_posthoc import B1_PROMPT, B2_PROMPT
        prompt = B1_PROMPT if method == "b1" else B2_PROMPT
        return "chat_prompt", prompt, hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()
    if method == "b4":
        from generate_b4_single_snapshot import SINGLE_SNAPSHOT_PROMPT
        prompt = SINGLE_SNAPSHOT_PROMPT
        return "chat_prompt", prompt, hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()
    raise ValueError(f"unknown method {method!r}")


async def _call_llm(method: str, llm_input: str, prompt_kind: str,
                    prompt_payload: str, model: str) -> str:
    if prompt_kind == "stored_prompt":
        from core.explanation_generation import generate_explanation_async
        return await generate_explanation_async(llm_input, prompt_payload)

    from openai import AsyncOpenAI
    client = AsyncOpenAI()
    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt_payload},
            {"role": "user", "content": llm_input},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


async def _call_llm_with_retry(method: str, llm_input: str, prompt_kind: str,
                               prompt_payload: str, model: str,
                               max_retries: int) -> str:
    delay = 1.0
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return await _call_llm(method, llm_input, prompt_kind, prompt_payload, model)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(delay + random.uniform(0.0, 0.5))
            delay *= 2
    assert last_exc is not None
    raise last_exc


def _source_records(records: List[Dict[str, Any]], query_ids: Optional[List[str]],
                    limit_groups: Optional[int]) -> List[Dict[str, Any]]:
    out = [r for r in records if isinstance(r, dict) and r.get("query_id")]
    if query_ids:
        qids = set(query_ids)
        out = [r for r in out if r.get("query_id") in qids]
    if limit_groups is not None:
        out = out[:limit_groups]
    return out


def _build_paraphrase_record(source: Dict[str, Any], para: Dict[str, Any],
                             method: str, prompt_sig: str,
                             model: str) -> Dict[str, Any]:
    qid = para["query_id"]
    rec = {
        "query_id": qid,
        "type_k": source.get("type_k"),
        "scenario": source.get("scenario"),
        "query_text": para["query_text"],
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "epoch": source.get("epoch"),
        "classification_label": source.get("classification_label"),
        "resolved_vehicle_targets": source.get("resolved_vehicle_targets", []),
        "pctl_formulas": source.get("pctl_formulas", []),
        "evidence": source.get("evidence", {}),
        "llm_input": _replace_query_line(source.get("llm_input") or "", para["query_text"]),
        "robustness_meta": {
            "method": method,
            "method_name": METHOD_NAMES[method],
            "source_query_id": source.get("query_id"),
            "base_id": para["base_id"],
            "paraphrase_id": para["paraphrase_id"],
            "evidence_source": "copied_from_source_record",
            "evidence_fixed": True,
            "prompt_signature": prompt_sig,
            "model": model,
        },
    }
    return rec


async def _generate_records(args: argparse.Namespace) -> int:
    _load_dotenv()
    input_path = args.input or DEFAULT_INPUTS[args.method]
    output_path = args.output or os.path.join(
        DEFAULT_OUTPUT_DIR, f"{args.method}_evidence.json"
    )

    labels = _read_labels(args.labels)
    queries = _read_queries(args.queries)
    groups = _paraphrase_groups(labels, queries)
    paraphrase_cache = _load_paraphrase_cache(args.paraphrase_cache)
    source = _source_records(_read_json(input_path), args.query_ids, args.limit_groups)
    if not source:
        print("No source records match filters; nothing to do.")
        return 0

    items: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for record in source:
        if args.paraphrase_source == "labels":
            label = labels.get(record.get("query_id"))
            if not label:
                print(f"[warn] no label row for {record.get('query_id')}; skipping")
                continue
            paras = groups.get(label["base_id"], [])
            if paras and paras[0]["query_text"] != record.get("query_text"):
                print(
                    f"[warn] labels/queries text for {record.get('query_id')} "
                    "does not match source evidence query_text; consider "
                    "--paraphrase-source generated"
                )
            if len(paras) < 2:
                print(f"[warn] base_id={label['base_id']} has <2 paraphrases; skipping")
                continue
        else:
            paras = await _generated_paraphrases_for_source(
                record, args, paraphrase_cache
            )
        for para in paras:
            items.append((record, para))

    full_order = [para["query_id"] for _, para in items]
    existing: List[Dict[str, Any]] = []
    done_ids = set()
    if args.resume and os.path.exists(output_path):
        existing = _load_existing(output_path)
        done_ids = {
            r.get("query_id") for r in existing
            if isinstance(r, dict) and r.get("status") == "ok"
        }
        items = [(r, p) for r, p in items if p["query_id"] not in done_ids]
        print(f"[resume] {len(done_ids)} done; {len(items)} remaining")

    if not items:
        print("Nothing to do.")
        return 0

    prompt_kind, prompt_payload, prompt_sig = _prompt_signature(args.method)
    records_by_id = {
        r["query_id"]: r for r in existing
        if isinstance(r, dict) and r.get("query_id") and r.get("status") == "ok"
    }
    semaphore = asyncio.Semaphore(max(1, args.workers))

    async def _bounded(source_record: Dict[str, Any], para: Dict[str, Any]) -> Dict[str, Any]:
        async with semaphore:
            t0 = time.time()
            rec = _build_paraphrase_record(
                source_record, para, args.method, prompt_sig, args.model
            )
            try:
                if (
                    int(para.get("paraphrase_id", -1)) == 0
                    and para.get("query_text") == source_record.get("query_text")
                    and source_record.get("explanation")
                ):
                    rec["explanation"] = source_record["explanation"]
                    rec["robustness_meta"]["explanation_source"] = (
                        "copied_original_explanation"
                    )
                elif args.skip_llm:
                    rec["explanation"] = None
                else:
                    task = _call_llm_with_retry(
                        args.method, rec["llm_input"], prompt_kind,
                        prompt_payload, args.model, args.max_retries
                    )
                    if args.timeout_sec > 0:
                        rec["explanation"] = await asyncio.wait_for(
                            task, timeout=args.timeout_sec
                        )
                    else:
                        rec["explanation"] = await task
                rec["status"] = "ok"
            except Exception:
                rec["status"] = "error"
                rec["error"] = traceback.format_exc()
                rec["explanation"] = None
            rec["elapsed_sec"] = round(time.time() - t0, 2)
            return rec

    print(
        f"Generating paraphrase robustness records: method={args.method} "
        f"source_groups={len(source)} records={len(items)} -> {output_path}"
    )
    print(
        f"workers={args.workers} skip_llm={args.skip_llm} "
        f"paraphrase_source={args.paraphrase_source} model={args.model}"
    )
    t_start = time.time()
    tasks = [asyncio.create_task(_bounded(r, p)) for r, p in items]
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
        _write_json_atomic(output_path, _ordered(records_by_id, full_order))

    print(
        f"\nDone. ok={n_ok} err={n_err} elapsed={time.time()-t_start:.1f}s "
        f"-> {output_path}"
    )
    return 0 if n_err == 0 else 1


def _to_float(value: Any) -> Optional[float]:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _norm_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _numeric_match(a: float, b: float, abs_tol: float, rel_tol: float) -> bool:
    diff = abs(a - b)
    rel = diff / max(abs(a), abs(b), 1.0)
    return diff <= abs_tol or rel <= rel_tol


def _dedupe_units(units: List[Tuple]) -> List[Tuple]:
    seen = set()
    out = []
    for unit in units:
        key = repr(unit)
        if key not in seen:
            out.append(unit)
            seen.add(key)
    return out


def _value_units(claim_record: Dict[str, Any]) -> List[Tuple[str, float]]:
    out = []
    for claim in claim_record.get("claims", []) or []:
        if claim.get("kind") != "numerical":
            continue
        f = _to_float(claim.get("value"))
        if f is not None:
            out.append(("num_value", f))
    return _dedupe_units(out)


def _claim_units(claim_record: Dict[str, Any]) -> List[Tuple]:
    out = []
    for claim in claim_record.get("claims", []) or []:
        kind = claim.get("kind")
        metric = claim.get("metric") or "unknown"
        subject = claim.get("subject") or "unspecified"
        snapshot = claim.get("snapshot") or "unspecified"
        if kind == "numerical":
            f = _to_float(claim.get("value"))
            if f is None:
                continue
            out.append(("numerical", metric, subject, snapshot, f))
        elif kind == "comparative":
            out.append((
                "comparative", metric, subject, snapshot,
                claim.get("direction") or "unknown",
            ))
        elif kind == "categorical":
            out.append((
                "categorical", metric, subject, snapshot,
                _norm_text(claim.get("value")),
            ))
        elif kind == "causal":
            out.append((
                "causal", metric, subject, snapshot,
                _norm_text(claim.get("text")),
            ))
    return _dedupe_units(out)


def _units_match(a: Tuple, b: Tuple, abs_tol: float, rel_tol: float) -> bool:
    if not a or not b or a[0] != b[0]:
        return False
    if a[0] == "num_value":
        return _numeric_match(a[1], b[1], abs_tol, rel_tol)
    if a[0] == "numerical":
        return (
            len(a) == len(b)
            and a[:4] == b[:4]
            and _numeric_match(a[4], b[4], abs_tol, rel_tol)
        )
    return a == b


def _fuzzy_jaccard(a_units: List[Tuple], b_units: List[Tuple],
                   abs_tol: float, rel_tol: float) -> float:
    a = list(a_units)
    b = list(b_units)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    used_b = set()
    matches = 0
    # Greedy matching is sufficient for these small sets and deterministic
    # after sorting by repr.
    for i, au in enumerate(sorted(a, key=repr)):
        for j, bu in enumerate(sorted(b, key=repr)):
            if j in used_b:
                continue
            if _units_match(au, bu, abs_tol, rel_tol):
                used_b.add(j)
                matches += 1
                break
    denom = len(a) + len(b) - matches
    return matches / denom if denom else 1.0


def _summary_stats(values: Sequence[float]) -> Dict[str, Any]:
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    vals_sorted = sorted(vals)
    return {
        "n": len(vals),
        "mean": sum(vals) / len(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "q25": vals_sorted[int(0.25 * (len(vals_sorted) - 1))],
        "q75": vals_sorted[int(0.75 * (len(vals_sorted) - 1))],
    }


def _score_groups(args: argparse.Namespace) -> int:
    evidence_records = _read_json(args.evidence)
    claim_records = _read_json(args.claims)
    if not isinstance(evidence_records, list) or not isinstance(claim_records, list):
        raise ValueError("evidence and claims must both be JSON arrays")

    labels = _read_labels(args.labels)
    evidence_by_id = {r.get("query_id"): r for r in evidence_records if isinstance(r, dict)}
    claims_by_id = {r.get("query_id"): r for r in claim_records if isinstance(r, dict)}

    grouped: Dict[str, List[str]] = defaultdict(list)
    for qid in evidence_by_id:
        rec = evidence_by_id[qid]
        meta = rec.get("robustness_meta") or {}
        base_id = meta.get("base_id")
        if not base_id and qid in labels:
            base_id = labels[qid].get("base_id")
        if base_id:
            grouped[base_id].append(qid)

    pairs = []
    by_type: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    by_scenario: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    for base_id, qids in sorted(grouped.items()):
        usable = [qid for qid in qids if qid in claims_by_id]
        if len(usable) < 2:
            continue
        def _para_order(qid: str) -> int:
            meta = (evidence_by_id.get(qid) or {}).get("robustness_meta") or {}
            if meta.get("paraphrase_id") is not None:
                return int(meta["paraphrase_id"])
            return int(labels.get(qid, {}).get("paraphrase_id", 0))

        usable = sorted(usable, key=_para_order)
        first = evidence_by_id[usable[0]]
        type_k = str(first.get("type_k"))
        scenario = str(first.get("scenario"))
        for qa, qb in combinations(usable, 2):
            ca = claims_by_id[qa]
            cb = claims_by_id[qb]
            numerical_j = _fuzzy_jaccard(
                _value_units(ca), _value_units(cb),
                args.abs_tol, args.rel_tol,
            )
            claim_j = _fuzzy_jaccard(
                _claim_units(ca), _claim_units(cb),
                args.abs_tol, args.rel_tol,
            )
            row = {
                "base_id": base_id,
                "query_id_a": qa,
                "query_id_b": qb,
                "type_k": type_k,
                "scenario": scenario,
                "numerical_value_jaccard": numerical_j,
                "claim_set_jaccard": claim_j,
            }
            pairs.append(row)
            by_type[type_k].append(row)
            by_scenario[scenario].append(row)

    summary = {
        "method": args.method,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "inputs": {
            "evidence": args.evidence,
            "claims": args.claims,
            "abs_tol": args.abs_tol,
            "rel_tol": args.rel_tol,
        },
        "overall": {
            "groups": len({p["base_id"] for p in pairs}),
            "pairs": len(pairs),
            "numerical_value_jaccard": _summary_stats(
                [p["numerical_value_jaccard"] for p in pairs]
            ),
            "claim_set_jaccard": _summary_stats(
                [p["claim_set_jaccard"] for p in pairs]
            ),
        },
        "by_type_k": {
            k: {
                "pairs": len(rows),
                "numerical_value_jaccard": _summary_stats(
                    [r["numerical_value_jaccard"] for r in rows]
                ),
                "claim_set_jaccard": _summary_stats(
                    [r["claim_set_jaccard"] for r in rows]
                ),
            }
            for k, rows in sorted(by_type.items(), key=lambda kv: int(kv[0]))
        },
        "by_scenario": {
            k: {
                "pairs": len(rows),
                "numerical_value_jaccard": _summary_stats(
                    [r["numerical_value_jaccard"] for r in rows]
                ),
                "claim_set_jaccard": _summary_stats(
                    [r["claim_set_jaccard"] for r in rows]
                ),
            }
            for k, rows in sorted(by_scenario.items(), key=lambda kv: int(kv[0]))
        },
        "pairs": pairs,
    }

    _write_json_atomic(args.output, summary)
    overall = summary["overall"]
    print("=== Paraphrase Robustness ===")
    print(f"method: {args.method}")
    print(f"groups: {overall['groups']} pairs: {overall['pairs']}")
    print(
        "numerical value Jaccard: "
        f"{overall['numerical_value_jaccard']['mean']:.3f}"
    )
    print(
        "claim-set Jaccard      : "
        f"{overall['claim_set_jaccard']['mean']:.3f}"
    )
    print(f"Output -> {args.output}")
    return 0


def _default_score_path(method: str) -> str:
    return os.path.join(DEFAULT_OUTPUT_DIR, f"{method}_summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate paraphrase explanations")
    gen.add_argument("--method", required=True, choices=["ours", "b1", "b2", "b4"])
    gen.add_argument("--input", help="Source method evidence JSON")
    gen.add_argument("--output", help="Output paraphrase evidence JSON")
    gen.add_argument("--labels", default=DEFAULT_LABELS)
    gen.add_argument("--queries", default=DEFAULT_QUERIES)
    gen.add_argument("--paraphrase-source", choices=["generated", "labels"],
                     default="generated",
                     help="Default generated uses current source query_text; labels uses legacy labels/queries files")
    gen.add_argument("--paraphrase-cache", default=DEFAULT_PARAPHRASE_CACHE,
                     help="Cache for generated paraphrase sets")
    gen.add_argument("--query-ids", nargs="+",
                     help="Source query_ids to expand into 5 paraphrases")
    gen.add_argument("--limit-groups", type=int,
                     help="Limit number of source query groups")
    gen.add_argument("--model", default=DEFAULT_MODEL)
    gen.add_argument("--workers", type=int, default=4)
    gen.add_argument("--timeout-sec", type=float, default=180.0)
    gen.add_argument("--max-retries", type=int, default=4)
    gen.add_argument("--skip-llm", action="store_true")
    gen.add_argument("--resume", action="store_true")

    score = sub.add_parser("score", help="Score extracted paraphrase claims")
    score.add_argument("--method", default="ours")
    score.add_argument("--evidence", required=True)
    score.add_argument("--claims", required=True)
    score.add_argument("--output")
    score.add_argument("--labels", default=DEFAULT_LABELS)
    score.add_argument("--abs-tol", type=float, default=0.5)
    score.add_argument("--rel-tol", type=float, default=0.02)
    args = parser.parse_args()
    if args.command == "score" and not args.output:
        args.output = _default_score_path(args.method)
    return args


def main() -> int:
    args = parse_args()
    if args.command == "generate":
        return asyncio.run(_generate_records(args))
    if args.command == "score":
        return _score_groups(args)
    raise ValueError(args.command)


if __name__ == "__main__":
    sys.exit(main())
