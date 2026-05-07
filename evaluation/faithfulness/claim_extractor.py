#!/usr/bin/env python3
"""
Stage-1 of faithfulness eval (Co-12 Dim 1A): decompose each NS explanation
into atomic claims.

Pipeline per explanation:
  Stage 1  regex pre-extract: numbers, vehicles, snapshot anchors
  Stage 2  ONE LLM call with hints + evidence vocab -> structured JSON list
  Stage 3  rule-based post-validation (substring, count, vocab, proximity)

Output: claims JSON — list of records, each with query_id + extracted claims.
The next script (claim_verifier.py) will read this + evidence_240.json and
emit V+/V-/U/N/A labels.

Usage:
    cd /home/yvyfz/Desktop/NS_XAI/ns_explainer

    # Pilot: 10 records
    python evaluation/faithfulness/claim_extractor.py --limit 10 \
        --output evaluation/faithfulness/claims_pilot.json

    # Full run, resumable
    python evaluation/faithfulness/claim_extractor.py \
        --output evaluation/faithfulness/claims_240.json --resume
"""

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))    # .../evaluation/faithfulness
EVAL_DIR = os.path.dirname(SCRIPT_DIR)                     # .../evaluation
REPO_DIR = os.path.dirname(EVAL_DIR)                       # .../ns_explainer
sys.path.insert(0, REPO_DIR)

# Load OPENAI_API_KEY from project-root .env (matches the rest of the repo).
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(REPO_DIR, ".env"))
except ImportError:
    pass

DEFAULT_INPUT = os.path.join(SCRIPT_DIR, "evidence_240.json")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "claims_240.json")

BASELINE_METRIC_VOCAB = [
    "BASELINE: ASSIGNED_VEHICLE",
    "BASELINE: CLOSEST_VEHICLE",
    "BASELINE: DISTANCE_TO_PICKUP_BY_VEHICLE",
    "BASELINE: PENDING_REQUESTS_BY_VEHICLE",
    "BASELINE: CURRENT_OCCUPANCY_BY_VEHICLE",
    "BASELINE: COMPLETED_REQUESTS_BY_VEHICLE",
    "BASELINE: VEHICLE_CAPACITY_BY_VEHICLE",
    "BASELINE: REQUEST_TIME",
    "BASELINE: EARLIEST_PICKUP",
    "BASELINE: LATEST_DROPOFF",
    "BASELINE: TRAFFIC_LEVEL",
    "BASELINE: POLICY_VISITS_BY_VEHICLE",
    "BASELINE: POLICY_VISIT_SHARE_BY_VEHICLE",
    "BASELINE: POLICY_Q_VALUE_BY_VEHICLE",
    "BASELINE: POLICY_TOTAL_VALUE_BY_VEHICLE",
]

# ---------- Vocab ----------

def load_metric_vocab() -> List[str]:
    """Union of all formulas across types in pa_pctl_mapping.PCTL_CTL_TEMPLATES."""
    from use_cases.paratransit.pa_pctl_mapping import PCTL_CTL_TEMPLATES
    vocab = set()
    for formulas in PCTL_CTL_TEMPLATES.values():
        for f in formulas:
            vocab.add(f)
    vocab.update(BASELINE_METRIC_VOCAB)
    return sorted(vocab)


# (scenario, request_num) -> (assigned_V_id, closest_V_id)
# Source: user-provided assignment specification per scenario+request.
# When closest == assigned the slot represents "alt is just the same vehicle"
# (no real second alternative for that request).
ASSIGNED_CLOSEST: Dict[Tuple[int, int], Tuple[int, int]] = {
    # S0 (CT) — Controlled Adaptation
    (0, 1): (4, 4), (0, 2): (4, 4), (0, 3): (4, 4), (0, 4): (4, 4),
    (0, 5): (0, 4), (0, 6): (3, 4),
    (0, 7): (0, 4), (0, 8): (0, 4), (0, 9): (0, 4), (0, 10): (0, 4),
    # S1 (AC) — Counter-intuitive (accident)
    (1, 1): (4, 0),
    (1, 2): (4, 1), (1, 3): (4, 1), (1, 4): (4, 1),
    (1, 5): (4, 3),
    (1, 6): (0, 0),
    (1, 7): (1, 2),
    (1, 8): (1, 3),
    (1, 9): (0, 0),
    (1, 10): (0, 3),
    # S2 (EV) — Event
    (2, 1): (4, 0),
    (2, 2): (3, 0),
    (2, 3): (4, 2),
    (2, 4): (3, 1),
    (2, 5): (2, 2),
    (2, 6): (0, 3),
    (2, 7): (0, 1),
    (2, 8): (1, 0),
    (2, 9): (1, 0),
    (2, 10): (3, 2),
}


def lookup_assigned_closest(scenario: int, request_num: Optional[int]
                            ) -> Tuple[Optional[int], Optional[int]]:
    """Return (assigned_v, closest_v) for a (scenario, request) pair, or
    (None, None) if request_num is None or no entry exists."""
    if request_num is None:
        return (None, None)
    return ASSIGNED_CLOSEST.get((scenario, request_num), (None, None))


# ---------- Stage 1: regex pre-extract ----------

# Match numbers like 82, 82.1, -0.5; reject 82.1.3 partial matches.
NUMBER_PATTERN = re.compile(r'(?<![\w.])(-?\d+(?:\.\d+)?)(?![\w.])')

# Words that, when preceding a number, mean the number is an *identifier*
# (vehicle/request/epoch index) rather than a metric VALUE. We drop such
# numbers from REGEX_NUMBERS so they don't pollute coverage stats and
# don't tempt the LLM into spurious numerical claims.
IDENTIFIER_PREFIX_PATTERN = re.compile(
    r'\b(vehicle|van|vehicles|vans|request|req|epoch|trip|stop|node|run|car)\s*$',
    re.I,
)

# V0..V9, "Vehicle 0".."Vehicle 9", "van 0".."van 9"
VEHICLE_PATTERN = re.compile(r'\bV(\d)\b|\bVehicle\s+(\d)\b|\bvan\s+(\d)\b', re.I)

# Snapshot anchor keywords. Lower-cased substring match.
SNAPSHOT_KEYWORDS = {
    "E_t": [
        "currently", "now", "after the crash", "after the accident",
        "after the change", "after the event", "after the road",
        "post-crash", "post-accident", "current model",
        "after the environment changed",
    ],
    "E_t_prev": [
        "before the environment changed", "before the crash",
        "before the event", "before the accident",
        "previously", "earlier", "had been", "had looked",
        "prior to the change", "prior to the crash",
    ],
    "delta": [
        "dropped from", "rose from", "shifted from",
        "changed from", "went from",
    ],
}


def extract_numbers(text: str) -> List[Tuple[str, int]]:
    """Return list of (number_string, position) for *value-bearing* numbers.

    Skips identifier numbers (e.g., "Vehicle 4", "request 2", "epoch 14") by
    checking up to 15 chars of preceding context for an identifier keyword.
    Also skips common PCTL threshold literals such as "60+ minute delay" and
    "15 minutes or less"; those are part of the proposition name, not cited
    evidence values.
    """
    out = []
    for m in NUMBER_PATTERN.finditer(text):
        start = m.start()
        end = m.end()
        prefix = text[max(0, start - 15):start]
        if IDENTIFIER_PREFIX_PATTERN.search(prefix):
            continue
        literal = m.group(1)
        try:
            f = float(literal)
        except ValueError:
            f = None
        if f is not None and abs(f - round(f)) < 1e-9 and int(round(f)) in {15, 30, 60}:
            left = text[max(0, start - 30):start].lower()
            right = text[end:end + 45].lower()
            threshold_context = (
                re.match(r"\s*\+\s*(?:min|minute)", right) or
                re.match(r"\s*(?:min|minute)s?\s*(?:or less|or more)", right) or
                "threshold" in left + right or
                re.search(r"(?:at least|less than|more than|under|over)\s*$", left)
            )
            if threshold_context:
                continue
        out.append((m.group(1), start))
    return out


def extract_vehicles(text: str) -> List[str]:
    out = set()
    for m in VEHICLE_PATTERN.finditer(text):
        for g in m.groups():
            if g:
                out.add(f"V{g}")
                break
    return sorted(out)


def extract_snapshot_anchors(text: str) -> List[Dict[str, str]]:
    found = []
    low = text.lower()
    for snap, kws in SNAPSHOT_KEYWORDS.items():
        for kw in kws:
            if kw in low:
                found.append({"keyword": kw, "snapshot": snap})
    return found


# ---------- Stage 2: LLM call ----------

def _get_client():
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    return OpenAI(api_key=api_key)


PROMPT_TEMPLATE = """You are extracting atomic factual claims from an AI
explanation about non-stationary paratransit dispatching. Each claim must
be (a) atomic — one fact per claim, (b) independently verifiable against
the planner's evidence, (c) a verbatim substring of the EXPLANATION.

For EACH atomic claim, output one JSON object with fields:
- text         : verbatim substring of the EXPLANATION (required)
- kind         : "numerical" | "categorical" | "comparative" | "causal"
- metric       : one of EVIDENCE_VOCAB, or "unknown"
- subject      : "V0".."V4" | "global" | "system" | "all" | "unspecified"
- snapshot     : "E_t" | "E_t_prev" | "delta" | "unspecified"
- value        : float (kind=numerical) | str (kind=categorical, e.g.
                 "bridge","none","busy","idle") | null
- direction    : "+" | "-" | "0" (only for kind=comparative; sign of
                 the directional comparison, else null)

Rules:
1. **MANDATORY numerical coverage**: For EVERY value in REGEX_NUMBERS,
   you MUST emit at least one claim with kind="numerical" whose `value`
   field equals that exact number. This is a hard requirement; missing
   any value will be flagged as an extraction error.

2. **Numerical claims are atomic and per-value**. If a sentence
   juxtaposes two values (e.g. "6.9 minutes for Vehicle 4 versus 16.9
   minutes for Vehicle 0"), you MUST emit TWO numerical claims —
   one for each value, each bound to its own subject — and OPTIONALLY
   one supplemental comparative claim on top. Never collapse two
   numerical values into a single comparative claim with value=null.
   Numerical first; comparative is supplemental.

3. Do NOT add numerical values that aren't in REGEX_NUMBERS. The
   REGEX_NUMBERS list is exhaustive of the value-bearing numbers in
   the explanation (identifier numbers like "Vehicle 4" are already
   excluded).

4. Skip vague qualitative statements like "the plan became less
   reliable" or "things looked worse" — only include claims with a
   concrete referent (a metric, a state, a directional comparison).

5. For comparative claims (e.g. "V4 was faster than V1"), set
   kind=comparative, value=null, direction reflecting the asserted
   ordering for the named subject vs the implicit other. These are
   SUPPLEMENTAL on top of the underlying numerical claims, not a
   replacement. Comparative claims must be on metrics that return
   numerical values (PCTL probabilities or DERIVED numerical metrics).
   For metrics that return strings (e.g., SERVICE_PATH_DISRUPTION returns
   "bridge"/"event"/"none"), use kind="categorical", never comparative.

6. For causal claims (e.g. "V4 was picked because risk was lower"),
   set kind=causal, value=null. These will not be verified numerically
   but are kept for completeness.

7. Use the SNAPSHOT_ANCHORS hints when binding snapshot. If unclear,
   use "unspecified".

8. **Snapshot label discipline (important)**:
   - Phrases like "moved from X to Y", "rose from X to Y", "X versus Y
     earlier", "X, up from Y", "X (down from Y)" describe two ABSOLUTE
     values at two snapshots. Emit TWO numerical claims: Y → snapshot
     "E_t_prev"; X → snapshot "E_t". DO NOT label either as "delta".
   - Use snapshot="delta" ONLY when the explanation explicitly states
     a difference / change magnitude itself, e.g. "increased by 24
     minutes", "the gap is 12 minutes", "delayed by 5%", "shorter by
     30 minutes". The numerical value in such cases IS the difference,
     not one of the two endpoints.

9. metric must come from EVIDENCE_VOCAB or "unknown". Do not invent
   new metric names.

10. Numbers that appear only as proposition thresholds (e.g., the "15"
    inside "dropoff_slack_le_15", or "30"/"60" inside "..._ge_30/60" or
    phrases like "60+ minute delay") describe the threshold, not the value
    of the metric. Do NOT extract such numbers as numerical claim values.
    The value of a P=? formula is its probability in [0, 1], often written
    in the explanation as a percentage such as "82.1%".

QUERY: {query}

EXPLANATION:
{explanation}

REGEX_NUMBERS (use only these numerical values, exactly): {numbers}
VEHICLES_MENTIONED:    {vehicles}
SNAPSHOT_ANCHORS:      {snapshot_anchors}
EVIDENCE_VOCAB:        {vocab}

Output a single JSON object:
{{"claims": [<claim_object>, <claim_object>, ...]}}

Do not output anything outside that JSON object."""


def call_extractor_llm(query: str, explanation: str,
                       numbers: List[str], vehicles: List[str],
                       anchors: List[Dict[str, str]],
                       vocab: List[str],
                       model: str = "gpt-5.4-mini",
                       temperature: float = 0.0) -> Dict[str, Any]:
    client = _get_client()
    prompt = PROMPT_TEMPLATE.format(
        query=query,
        explanation=explanation,
        numbers=numbers,
        vehicles=vehicles,
        snapshot_anchors=anchors,
        vocab=vocab,
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=temperature,
    )
    txt = resp.choices[0].message.content
    return json.loads(txt)


# ---------- Stage 3: rule post-validate ----------

VALID_KINDS = {"numerical", "categorical", "comparative", "causal"}
VALID_SUBJECTS = {"V0", "V1", "V2", "V3", "V4",
                  "global", "system", "all", "unspecified"}
VALID_SNAPSHOTS = {"E_t", "E_t_prev", "delta", "unspecified"}


def _to_float(x: Any):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# Normalize markdown emphasis + whitespace so the LLM's paraphrase-tolerant
# rendering of `text` (e.g. dropping `**bold**` markers) still matches against
# the markdown-bearing explanation. This is purely for substring validation.
_MD_PATTERN = re.compile(r'\*+|_+|`+')
_WS_PATTERN = re.compile(r'\s+')


def _normalize_for_substring(s: str) -> str:
    s = _MD_PATTERN.sub('', s)
    s = _WS_PATTERN.sub(' ', s).strip()
    return s.lower()


# Sentence boundary: ., !, ? followed by whitespace/end-of-string AND not
# followed by a digit (so "0.5 minutes" stays whole — period is followed by
# a digit). The lookahead is on the right side: "V1. But..." has period
# followed by space → boundary; "0.5" has period followed by digit → not.
# Newlines also act as boundaries (LLM explanations often use blank lines).
_SENTENCE_BOUNDARY = re.compile(r'[.!?]+(?!\d)(?:\s+|$)|\n+')


def _sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return [(start, end)] of each sentence in `text`."""
    cuts = [0]
    for m in _SENTENCE_BOUNDARY.finditer(text):
        cuts.append(m.end())
    if cuts[-1] < len(text):
        cuts.append(len(text))
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)
            if cuts[i] < cuts[i + 1]]


def _which_sentence(pos: int, spans: List[Tuple[int, int]]) -> Optional[int]:
    """Return the index of the sentence span containing char position `pos`,
    or None if pos is out of all spans."""
    for i, (s, e) in enumerate(spans):
        if s <= pos < e:
            return i
    return None


# Pronoun phrases that, given a (scenario, request) lookup, can be resolved to
# a specific vehicle. The first capture group is the role keyword.
_CLOSEST_PRONOUN_PATTERN = re.compile(
    r"\b(?:the|a|its)\s+(?:closest|nearest)\s+(?:vehicle|van|car|one)\b",
    re.I,
)
_ASSIGNED_PRONOUN_PATTERN = re.compile(
    r"\b(?:the|its|this)\s+(?:assigned|chosen|picked|selected)\s+(?:vehicle|van|car|one)\b",
    re.I,
)


def _vehicle_positions_with_pronouns(text: str,
                                     assigned_v: Optional[int],
                                     closest_v: Optional[int]
                                     ) -> Dict[str, List[int]]:
    """Return positions of all vehicle mentions per V-id (V0..V4), including
    pronoun resolutions for "the closest vehicle" -> closest_v and "the
    assigned vehicle" -> assigned_v.
    """
    positions: Dict[str, List[int]] = {f"V{i}": [] for i in range(5)}

    # Direct mentions: V0..V9, "Vehicle 0", "van 0"
    for m in VEHICLE_PATTERN.finditer(text):
        for g in m.groups():
            if g:
                key = f"V{g}"
                if key in positions:
                    positions[key].append(m.start())
                break

    # Pronouns
    if closest_v is not None:
        for m in _CLOSEST_PRONOUN_PATTERN.finditer(text):
            positions[f"V{closest_v}"].append(m.start())
    if assigned_v is not None:
        for m in _ASSIGNED_PRONOUN_PATTERN.finditer(text):
            positions[f"V{assigned_v}"].append(m.start())

    # Sort positions per vehicle so callers can do nearest-mention reasoning
    for k in positions:
        positions[k].sort()
    return positions


def validate_claims(claims: List[Dict[str, Any]], explanation: str,
                    regex_numbers: List[Tuple[str, int]],
                    vocab: List[str],
                    assigned_v: Optional[int] = None,
                    closest_v: Optional[int] = None,
                    sentence_buffer: int = 1,
                    enable_implicit_topic: bool = False
                    ) -> Tuple[List[Dict[str, Any]], List[Tuple[Dict[str, Any], str]]]:
    """Filter / annotate claims. Returns (kept, dropped_with_reason).

    Numerical-claim validation uses sentence-level proximity:
      1. value must be in regex set (no fabricated numbers)
      2. subject's mention(s) — direct ("V4") or pronoun ("the closest
         vehicle" resolved via assigned_v / closest_v) — must appear in
         the same sentence as the value, OR within ±sentence_buffer
         sentences of it.

    If `enable_implicit_topic` is True, a fallback path also accepts a
    binding when the subject is the *most recent* vehicle mention strictly
    before the value's position (handles narrative / markdown-section
    explanations where the subject is named once at the top of a section
    and not repeated near each datum). Recommended for type 15 only.
    """
    kept = []
    dropped = []
    vocab_set = set(vocab)
    norm_explanation = _normalize_for_substring(explanation)
    sent_spans = _sentence_spans(explanation)
    veh_positions = _vehicle_positions_with_pronouns(explanation,
                                                     assigned_v, closest_v)

    # Build lookup: float-rounded value -> literal source strings in regex
    # set (so we can find their positions in explanation).
    regex_value_lookup: Dict[float, List[str]] = {}
    for s, _ in regex_numbers:
        f = _to_float(s)
        if f is not None:
            regex_value_lookup.setdefault(round(f, 4), []).append(s)

    def _value_positions(value_str: str) -> List[int]:
        pat = re.compile(rf'(?<![\w.]){re.escape(value_str)}(?![\w.])')
        return [m.start() for m in pat.finditer(explanation)]

    # All vehicle mentions across all V-ids, sorted by position. Used by
    # the implicit-topic fallback (only triggered when enabled).
    all_mentions_sorted: List[Tuple[int, str]] = sorted(
        (p, v) for v, ps in veh_positions.items() for p in ps
    )

    def _subject_in_sentence_window(subj: str, value_str: str) -> bool:
        """True iff `value_str` is bound to `subj`. Always tries Pass 1
        (sentence-level proximity ±sentence_buffer). If `enable_implicit_topic`
        is True, also tries Pass 2 (latest-vehicle-mention-before-value).
        """
        subj_positions = veh_positions.get(subj, [])
        if not subj_positions:
            return False

        value_positions = _value_positions(value_str)

        # ---- Pass 1: sentence-level proximity ----
        subj_sent_idxs = {_which_sentence(p, sent_spans) for p in subj_positions}
        subj_sent_idxs.discard(None)
        if subj_sent_idxs:
            for vp in value_positions:
                v_idx = _which_sentence(vp, sent_spans)
                if v_idx is None:
                    continue
                if any(abs(s_idx - v_idx) <= sentence_buffer
                       for s_idx in subj_sent_idxs):
                    return True

        # ---- Pass 2 (opt-in): implicit-topic fallback ----
        # For each value position, find the latest vehicle mention strictly
        # before it. If that mention's vehicle equals `subj`, accept.
        if enable_implicit_topic:
            for vp in value_positions:
                latest_v = None
                for p, v in all_mentions_sorted:
                    if p < vp:
                        latest_v = v
                    else:
                        break
                if latest_v == subj:
                    return True

        return False

    for c in claims:
        if not isinstance(c, dict):
            dropped.append(({}, f"non-dict claim: {c!r}")); continue

        # Required: text (kept as traceability metadata; not a hard gate
        # for numerical claims, but still required structurally)
        text = c.get("text")
        if not isinstance(text, str) or not text.strip():
            dropped.append((c, "missing/empty text")); continue

        # kind
        if c.get("kind") not in VALID_KINDS:
            dropped.append((c, f"bad kind: {c.get('kind')}")); continue

        # snapshot — coerce out-of-vocab to "unspecified"
        if c.get("snapshot") not in VALID_SNAPSHOTS:
            c["snapshot"] = "unspecified"

        # subject — coerce to "unspecified" if invalid
        if c.get("subject") not in VALID_SUBJECTS:
            c.setdefault("subject_warning", c.get("subject"))
            c["subject"] = "unspecified"

        # metric — flag if not in vocab (don't drop)
        metric = c.get("metric")
        if metric and metric != "unknown" and metric not in vocab_set:
            c["metric_warning"] = "not in vocab"

        # ---- per-kind validation ----
        if c["kind"] == "numerical":
            # 1. Value must be a valid float and present in the regex set
            v = _to_float(c.get("value"))
            if v is None:
                dropped.append((c, "numerical claim missing/invalid value")); continue
            v_key = round(v, 4)
            if v_key not in regex_value_lookup:
                dropped.append((c, f"value {v} not in regex set")); continue

            # 2. Subject must share (or be within ±sentence_buffer of) the
            #    value's sentence. Pronouns ("the closest vehicle") count
            #    as virtual mentions of the resolved V-id.
            subj = c.get("subject")
            if isinstance(subj, str) and subj.startswith("V"):
                hit = any(_subject_in_sentence_window(subj, s)
                          for s in regex_value_lookup[v_key])
                if not hit:
                    dropped.append((c,
                        f"subject {subj} not in same sentence (±{sentence_buffer}) "
                        f"as value {v} in explanation")); continue
            # subject is "global" / "system" / "unspecified" → no proximity gate

            # Soft note: text is not a substring (informational only)
            if text not in explanation and \
               _normalize_for_substring(text) not in norm_explanation:
                c["text_substring_warning"] = "claim.text not a substring of explanation"

        else:
            # Non-numerical: text must still be (loosely) traceable to
            # the explanation, since there's no value to anchor on.
            if text not in explanation and \
               _normalize_for_substring(text) not in norm_explanation:
                dropped.append((c, "text not a substring (even after norm)")); continue

        kept.append(c)

    return kept, dropped


def coverage_check(kept: List[Dict[str, Any]],
                   regex_numbers: List[Tuple[str, int]]) -> Dict[str, Any]:
    """How many regex numbers got at least one numerical claim?"""
    seen = set()
    for c in kept:
        if c.get("kind") == "numerical":
            v = _to_float(c.get("value"))
            if v is not None:
                seen.add(round(v, 4))
    regex_floats = {round(_to_float(s), 4) for s, _ in regex_numbers
                    if _to_float(s) is not None}
    return {
        "regex_numbers": sorted(regex_floats),
        "covered_by_claims": sorted(regex_floats & seen),
        "missed_numbers": sorted(regex_floats - seen),
    }


# ---------- Pipeline ----------

def process_record(record: Dict[str, Any], vocab: List[str],
                   model: str) -> Dict[str, Any]:
    qid = record["query_id"]
    expl = record.get("explanation") or ""
    if not expl:
        return {"query_id": qid, "status": "no_explanation", "claims": []}

    query = record.get("query_text", "")
    scenario = record.get("scenario")
    numbers = extract_numbers(expl)
    vehicles = extract_vehicles(expl)
    anchors = extract_snapshot_anchors(expl)

    # Resolve (assigned, closest) from the (scenario, request) lookup so that
    # pronouns like "the closest vehicle" can be treated as virtual mentions
    # of the right V during sentence-level proximity validation.
    request_num: Optional[int] = None
    assigned_v: Optional[int] = None
    closest_v: Optional[int] = None
    try:
        from use_cases.paratransit.config import ParatransitConfig
        # Pass scenario_type=None so the raw 1..10 number is returned (no
        # user-study display↔original reverse mapping).
        request_num = ParatransitConfig.extract_epoch_from_query(query, None)
    except Exception:
        request_num = None
    if scenario is not None and request_num is not None:
        assigned_v, closest_v = lookup_assigned_closest(int(scenario), int(request_num))

    try:
        raw = call_extractor_llm(
            query=query, explanation=expl,
            numbers=[s for s, _ in numbers], vehicles=vehicles,
            anchors=anchors, vocab=vocab, model=model,
        )
    except Exception as e:
        return {"query_id": qid, "status": "llm_error",
                "error": f"{type(e).__name__}: {e}", "claims": []}

    raw_claims = raw.get("claims", [])
    if not isinstance(raw_claims, list):
        return {"query_id": qid, "status": "bad_llm_output",
                "error": "claims is not a list", "claims": []}

    # Type 15 ("how much does congestion affect assigned timing?") explanations
    # use a multi-section markdown structure where the subject vehicle is named
    # once at the top and not repeated near each datum. Enable implicit-topic
    # fallback only for that family — other types' compact structures don't
    # need it and tightening the gate elsewhere preserves swap-detection.
    type_k = record.get("type_k")
    enable_implicit = (type_k == 15)

    kept, dropped = validate_claims(raw_claims, expl, numbers, vocab,
                                    assigned_v=assigned_v, closest_v=closest_v,
                                    enable_implicit_topic=enable_implicit)
    cov = coverage_check(kept, numbers)

    return {
        "query_id": qid,
        "status": "ok",
        "extraction_meta": {
            "regex_numbers": [s for s, _ in numbers],
            "vehicles": vehicles,
            "snapshot_anchors": anchors,
            "request_num": request_num,
            "assigned_v": assigned_v,
            "closest_v": closest_v,
            "n_raw": len(raw_claims),
            "n_kept": len(kept),
            "n_dropped": len(dropped),
            "drop_reasons": [{"reason": r,
                              "claim_text": (c.get("text") if isinstance(c, dict) else str(c))[:120]}
                             for c, r in dropped],
            "coverage": cov,
        },
        "claims": kept,
    }


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=DEFAULT_INPUT,
                   help="Input evidence JSON (from generate_evidence.py)")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="Output claims JSON")
    p.add_argument("--query-ids", nargs="+",
                   help="Run only these query_ids (default: all)")
    p.add_argument("--limit", type=int, help="Cap N records (after filtering)")
    p.add_argument("--model", default="gpt-5.4-mini",
                   help="OpenAI model id. (default: gpt-5.4-mini)")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel LLM calls (default 4)")
    p.add_argument("--resume", action="store_true",
                   help="Skip query_ids already present in --output")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.input) as f:
        records = json.load(f)
    if args.query_ids:
        qid_set = set(args.query_ids)
        records = [r for r in records if r["query_id"] in qid_set]
    if args.limit is not None:
        records = records[:args.limit]

    existing: Dict[str, Dict[str, Any]] = {}
    if args.resume and os.path.exists(args.output):
        with open(args.output) as f:
            try:
                for r in json.load(f):
                    if r.get("status") == "ok":
                        existing[r["query_id"]] = r
            except Exception as e:
                print(f"[warn] could not parse existing output: {e}; starting fresh")
                existing = {}
    if args.resume:
        records = [r for r in records if r["query_id"] not in existing]
        print(f"[resume] {len(existing)} ok records done; {len(records)} remaining")

    if not records:
        print("Nothing to do.")
        return 0

    vocab = load_metric_vocab()
    print(f"Loaded {len(vocab)} formula vocab entries; using model={args.model}")

    all_out: List[Dict[str, Any]] = list(existing.values())

    def _flush():
        tmp = args.output + ".tmp"
        with open(tmp, "w") as f:
            json.dump(all_out, f, indent=2, ensure_ascii=False)
        os.replace(tmp, args.output)

    t0 = time.time()
    n_ok = n_err = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_record, r, vocab, args.model): r for r in records}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                out = fut.result()
            except Exception as e:
                rec = futs[fut]
                out = {"query_id": rec["query_id"], "status": "fatal",
                       "error": f"{type(e).__name__}: {e}", "claims": []}
            all_out.append(out)
            ok = out["status"] == "ok"
            n_ok += int(ok); n_err += int(not ok)
            cov = out.get("extraction_meta", {}).get("coverage", {})
            n_regex = len(cov.get("regex_numbers", []))
            n_cov = len(cov.get("covered_by_claims", []))
            mark = "OK " if ok else "ERR"
            print(f"[{i}/{len(records)}] {mark} {out['query_id']} "
                  f"claims={len(out.get('claims', []))} num_cov={n_cov}/{n_regex}")
            if i % 5 == 0:
                _flush()

    _flush()
    print(f"\nDone. ok={n_ok} err={n_err} elapsed={time.time()-t0:.1f}s "
          f"-> {args.output}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
