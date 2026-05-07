#!/usr/bin/env python3
"""
Generate current-benchmark query classification files.

This replaces the legacy query-classification TSVs with files aligned to
evaluation/core_queries_240.csv:
  - evaluation/queries.txt
  - evaluation/labels.txt

By default, the output contains exactly the original 240 query prompts used by
the explanation benchmark. Pass --include-paraphrases to additionally generate
four paraphrases per query for robustness checks.
"""

import argparse
import asyncio
import csv
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.dirname(HERE)
REPO_DIR = os.path.dirname(EVAL_DIR)
sys.path.insert(0, REPO_DIR)

DEFAULT_CORE = os.path.join(EVAL_DIR, "core_queries_240.csv")
DEFAULT_QUERIES = os.path.join(EVAL_DIR, "queries.txt")
DEFAULT_LABELS = os.path.join(EVAL_DIR, "labels.txt")
DEFAULT_CACHE = os.path.join(HERE, "current_paraphrases.json")
DEFAULT_MODEL = "gpt-5.4-mini"

SCENARIO_LABELS = {
    "0": "CT",
    "1": "AC",
    "2": "EV",
}

PARAPHRASE_PROMPT = """Generate four paraphrases of the dispatch query.

Requirements:
- Preserve the exact meaning and query type.
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


def _read_core(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_cache(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"items": {}}
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    data.setdefault("items", {})
    return data


def _write_json_atomic(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _flush_cache(path: str, cache: Dict[str, Any]) -> None:
    cache["updated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _write_json_atomic(path, cache)


async def _call_paraphrase_llm(query: str, model: str) -> List[str]:
    from openai import AsyncOpenAI

    client = AsyncOpenAI()
    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": PARAPHRASE_PROMPT.format(query=query)}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    paras = data.get("paraphrases")
    if not isinstance(paras, list):
        raise ValueError("response missing JSON list field 'paraphrases'")
    cleaned = [p.strip() for p in paras if isinstance(p, str) and p.strip()]
    if len(cleaned) < 4:
        raise ValueError(f"expected 4 paraphrases, got {len(cleaned)}")
    return cleaned[:4]


def _fallback_paraphrases(query: str) -> List[str]:
    if not query:
        return ["", "", "", ""]
    lower_first = query[0].lower() + query[1:]
    return [
        f"Can you explain this dispatch decision: {query}",
        f"In operational terms, {lower_first}",
        f"What is the reason behind this: {query}",
        f"Could you clarify the dispatch rationale here: {query}",
    ]


async def _ensure_paraphrases(row: Dict[str, str], args: argparse.Namespace,
                              cache: Dict[str, Any]) -> Dict[str, Any]:
    qid = row["query_id"]
    query = row["query_text"]
    if not args.include_paraphrases:
        return {
            "query_id": qid,
            "type_k": int(row["type_k"]),
            "scenario": int(row["scenario"]),
            "source_query_text": query,
            "paraphrases": [query],
            "source": "original_benchmark_query",
        }

    cached = cache.get("items", {}).get(qid)
    if isinstance(cached, dict) and isinstance(cached.get("paraphrases"), list):
        return cached

    if args.skip_llm:
        generated = _fallback_paraphrases(query)
        source = "deterministic_fallback"
    else:
        generated = await _call_paraphrase_llm(query, args.model)
        source = "llm_generated"

    item = {
        "query_id": qid,
        "type_k": int(row["type_k"]),
        "scenario": int(row["scenario"]),
        "source_query_text": query,
        "paraphrases": [query] + generated,
        "source": source,
    }
    cache.setdefault("items", {})[qid] = item
    _flush_cache(args.cache, cache)
    return item


def _write_tsvs(items: List[Dict[str, Any]], queries_path: str,
                labels_path: str, include_paraphrases: bool) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(queries_path)) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(labels_path)) or ".", exist_ok=True)

    q_tmp = queries_path + ".tmp"
    l_tmp = labels_path + ".tmp"
    with open(q_tmp, "w", encoding="utf-8", newline="") as qf, \
            open(l_tmp, "w", encoding="utf-8", newline="") as lf:
        qw = csv.writer(qf, delimiter="\t")
        lw = csv.writer(lf, delimiter="\t")
        qw.writerow(["id", "query"])
        lw.writerow([
            "id", "type_k", "base_id", "paraphrase_id",
            "scenario", "is_adversarial", "adversarial_subtype",
        ])
        for item in items:
            base_id = item["query_id"]
            scenario = SCENARIO_LABELS.get(str(item["scenario"]), str(item["scenario"]))
            queries = item["paraphrases"] if include_paraphrases else item["paraphrases"][:1]
            for idx, query in enumerate(queries):
                qid = base_id if idx == 0 else f"{base_id}_p{idx}"
                qw.writerow([qid, query])
                lw.writerow([qid, item["type_k"], base_id, idx, scenario, 0, ""])
    os.replace(q_tmp, queries_path)
    os.replace(l_tmp, labels_path)


async def _run(args: argparse.Namespace) -> int:
    _load_dotenv()
    rows = _read_core(args.core)
    if args.query_ids:
        qids = set(args.query_ids)
        rows = [r for r in rows if r["query_id"] in qids]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        print("No rows match filters; nothing to do.")
        return 0

    cache = _read_cache(args.cache)
    semaphore = asyncio.Semaphore(max(1, args.workers))
    results: Dict[str, Dict[str, Any]] = {}

    async def _bounded(row: Dict[str, str]) -> Dict[str, Any]:
        async with semaphore:
            t0 = time.time()
            try:
                item = await _ensure_paraphrases(row, args, cache)
                item["status"] = "ok"
            except Exception:
                item = {
                    "query_id": row["query_id"],
                    "type_k": int(row["type_k"]),
                    "scenario": int(row["scenario"]),
                    "source_query_text": row["query_text"],
                    "paraphrases": [row["query_text"]] + _fallback_paraphrases(row["query_text"]),
                    "status": "error",
                    "error": traceback.format_exc(),
                }
            item["elapsed_sec"] = round(time.time() - t0, 2)
            return item

    print(
        f"Generating current-aligned query files from {args.core}\n"
        f"rows={len(rows)} workers={args.workers} "
        f"include_paraphrases={args.include_paraphrases} "
        f"skip_llm={args.skip_llm} model={args.model}"
    )
    tasks = [asyncio.create_task(_bounded(r)) for r in rows]
    n_ok = n_err = 0
    for i, task in enumerate(asyncio.as_completed(tasks), 1):
        item = await task
        results[item["query_id"]] = item
        ok = item.get("status") == "ok"
        n_ok += int(ok)
        n_err += int(not ok)
        mark = "OK " if ok else "ERR"
        print(f"[{i}/{len(tasks)}] {mark} {item['query_id']} elapsed={item['elapsed_sec']}s")

    ordered = [results[r["query_id"]] for r in rows]
    _write_tsvs(ordered, args.queries_out, args.labels_out, args.include_paraphrases)
    _write_json_atomic(args.cache, cache)

    print(f"\nDone. ok={n_ok} err={n_err}")
    print(f"Queries -> {args.queries_out}")
    print(f"Labels  -> {args.labels_out}")
    print(f"Cache   -> {args.cache}")
    return 0 if n_err == 0 else 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core", default=DEFAULT_CORE)
    p.add_argument("--queries-out", default=DEFAULT_QUERIES)
    p.add_argument("--labels-out", default=DEFAULT_LABELS)
    p.add_argument("--cache", default=DEFAULT_CACHE)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--query-ids", nargs="+")
    p.add_argument("--include-paraphrases", action="store_true",
                   help="Write original query plus four paraphrases per query. "
                        "Default writes exactly the original benchmark queries.")
    p.add_argument("--skip-llm", action="store_true",
                   help="Use deterministic fallback paraphrases; for smoke tests only")
    return p.parse_args()


def main() -> int:
    return asyncio.run(_run(parse_args()))


if __name__ == "__main__":
    sys.exit(main())
