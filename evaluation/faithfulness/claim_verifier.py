#!/usr/bin/env python3
"""
Stage-2 of faithfulness eval (Co-12 Dim 1A): verify each atomic claim
against ground-truth PCTL/derived evidence and assign V+/V-/U.

Per-claim logic by kind:
  numerical   : look up evidence[snapshot][metric][subject], tolerance-compare
                claim.value vs gt -> V+/V-/U.
  categorical : look up evidence[snapshot][metric][subject], string-equality
                claim.value vs gt -> V+/V-/U.
  comparative : compute the actual sign of (subject - comparator) at the
                stated snapshot, or sign(E_t - E_t_prev) for snapshot=delta;
                comparator is the (assigned, closest) counterpart of subject.
  causal      : not graded ("N/A"), kept in output for traceability.

Inputs:
  --claims     evaluation/faithfulness/claims_240.json
  --evidence   evaluation/faithfulness/evidence_240.json
  --output     evaluation/faithfulness/graded_claims_240.json

Outputs:
  Per-claim record with verdict + reason + gt_value.
  Aggregate report (V+/V-/U rates per type_k, per scenario, overall).
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # .../evaluation/faithfulness
EVAL_DIR = os.path.dirname(SCRIPT_DIR)                     # .../evaluation
REPO_DIR = os.path.dirname(EVAL_DIR)                       # .../ns_explainer

DEFAULT_CLAIMS = os.path.join(SCRIPT_DIR, "claims_240.json")
DEFAULT_EVIDENCE = os.path.join(SCRIPT_DIR, "evidence_240.json")
DEFAULT_OUTPUT = os.path.join(SCRIPT_DIR, "graded_claims_240.json")
DEFAULT_QUERY_CSV = os.path.join(EVAL_DIR, "core_queries_240.csv")


# ---------- helpers ----------

def _to_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _canonical_metric(metric: Any) -> Any:
    # Some extractor outputs omit the DERIVED prefix despite the vocabulary
    # using DERIVED: STEP_CONFIDENCE.
    if metric == "STEP_CONFIDENCE":
        return "DERIVED: STEP_CONFIDENCE"
    return metric


_VEHICLE_ID_RE = re.compile(r"^V[0-4]$")
_VEHICLE_MENTION_RE = re.compile(r"\bV([0-4])\b|\bVehicle\s+([0-4])\b|\bvan\s+([0-4])\b", re.I)
_NUMBER_RE = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])")


def _is_vehicle_id(x: Any) -> bool:
    return isinstance(x, str) and _VEHICLE_ID_RE.match(x) is not None


def _claim_text(claim: Dict[str, Any]) -> str:
    return str(claim.get("text") or "")


def _vehicles_in_text(text: str) -> List[str]:
    out = []
    for m in _VEHICLE_MENTION_RE.finditer(text or ""):
        for g in m.groups():
            if g is not None:
                v = f"V{g}"
                if v not in out:
                    out.append(v)
                break
    return out


def _vehicle_items(metric_value: Any,
                   only: Optional[List[str]] = None) -> List[Tuple[str, Any]]:
    if not isinstance(metric_value, dict):
        return []
    allowed = set(only or [])
    out = []
    for k, v in sorted(metric_value.items()):
        if _is_vehicle_id(k) and (not allowed or k in allowed):
            out.append((k, v))
    return out


def _snapshots_for(snapshot: Optional[str]) -> List[str]:
    if snapshot in ("E_t", "E_t_prev"):
        return [snapshot]
    if snapshot in (None, "unspecified"):
        return ["E_t", "E_t_prev"]
    # Some extractor outputs label "request 5-6 interval" as delta. For
    # nested interval evidence, the value is stored at both snapshots.
    if snapshot == "delta":
        return ["E_t"]
    return []


def _metric_raw(evidence: Dict[str, Any], snapshot: str, metric: str) -> Any:
    snap = evidence.get(snapshot)
    if not isinstance(snap, dict):
        return None
    return snap.get(metric)


def _lookup_evidence(evidence: Dict[str, Any], snapshot: str, metric: str,
                     subject: Optional[str]) -> Any:
    """Return evidence[snapshot][metric][subject], or evidence[snapshot][metric]
    when the metric is a scalar (e.g. TRAFFIC_LEVEL). None if anything missing."""
    snap = evidence.get(snapshot)
    if not isinstance(snap, dict):
        return None
    m = snap.get(metric)
    if m is None:
        return None
    if isinstance(m, dict):
        if isinstance(subject, str) and subject.startswith("V"):
            return m.get(subject)
        return None
    return m


def _looks_universal(text: str) -> bool:
    low = (text or "").lower()
    return any(tok in low for tok in (
        "all vehicles", "every vehicle", "each vehicle", "both vehicles",
        "for every vehicle", "for all vehicles", "for both", "all have",
        "both are", "each has", "all are",
    ))


def _looks_like_range(text: str) -> bool:
    low = (text or "").lower()
    return bool(re.search(r"\b(?:range|between|from)\b", low) or
                re.search(r"\d(?:\.\d+)?\s*(?:-|to|–)\s*-?\d", low))


def _threshold_literal(value: float, text: str, metric: str) -> bool:
    """True when a number is the threshold inside a thresholded metric
    ("60+ minute delay", "15 minutes or less"), not a metric value."""
    if abs(value - round(value)) > 1e-9:
        return False
    if int(round(value)) not in {15, 30, 60}:
        return False
    low = (text or "").lower()
    if re.search(rf"\b{int(value)}\s*\+\s*(?:min|minute)", low):
        return True
    if re.search(rf"\b{int(value)}\s*(?:min|minute)s?\s*(?:or less|or more)", low):
        return True
    if "threshold" in low and str(int(value)) in low:
        return True
    if re.search(rf"(?:at least|less than|more than|under|over)\s+{int(value)}\s*(?:min|minute)", low):
        return True
    return isinstance(metric, str) and (
        f"_ge_{int(value)}" in metric or f"_le_{int(value)}" in metric
    ) and str(int(value)) in low


def _interval_numeric_candidates(raw: Any) -> List[Tuple[str, float]]:
    """Flatten nested TURNING_INTERVAL evidence into checkable numbers."""
    vals: List[Tuple[str, float]] = []
    if isinstance(raw, dict):
        for key in ("display_interval", "recovery_request_interval"):
            seq = raw.get(key)
            if isinstance(seq, list):
                for i, v in enumerate(seq):
                    f = _to_float(v)
                    if f is not None:
                        vals.append((f"{key}[{i}]", f))
        rec = raw.get("recovery_interval")
        if isinstance(rec, dict):
            for key in ("start_epoch", "end_epoch"):
                f = _to_float(rec.get(key))
                if f is not None:
                    vals.append((f"recovery_interval.{key}", f))
        return vals
    f = _to_float(raw)
    return [("value", f)] if f is not None else []


def _interval_claim_matches(raw: Any, text_or_value: str) -> bool:
    nums = [_to_float(m.group(0)) for m in _NUMBER_RE.finditer(text_or_value or "")]
    nums = [n for n in nums if n is not None]
    if not nums:
        return False
    vals = [v for _, v in _interval_numeric_candidates(raw)]
    if len(nums) >= 2:
        # Interval claims usually say "5-6", "between requests 3 and 4", etc.
        claim_pair = {int(round(nums[-2])), int(round(nums[-1]))}
        for i in range(len(vals) - 1):
            if {int(round(vals[i])), int(round(vals[i + 1]))} == claim_pair:
                return True
    return any(abs(n - v) <= 0.5 for n in nums for v in vals)


def _numeric_match(value: float, gt: float, text: str = "",
                   abs_tol: float = 0.5, rel_tol: float = 0.02,
                   allow_signed_magnitude: bool = False) -> Tuple[bool, str]:
    diff = abs(value - gt)
    rel = diff / max(abs(gt), 1.0)
    if diff <= abs_tol or rel <= rel_tol:
        return True, "direct"

    # Evidence probabilities are stored as 0..1, while explanations often
    # cite percentages. This applies to PCTL formulas and derived probability
    # metrics such as STEP_CONFIDENCE.
    if 0.0 <= gt <= 1.0:
        scaled = value / 100.0
        if abs(scaled - gt) <= max(rel_tol * abs(gt), abs_tol / 100.0):
            return True, "percent"
        if abs(value * 100.0 - gt) <= max(rel_tol * abs(gt), abs_tol / 100.0):
            return True, "fraction"

    if allow_signed_magnitude:
        mag_diff = abs(abs(value) - abs(gt))
        mag_rel = mag_diff / max(abs(gt), 1.0)
        if mag_diff <= abs_tol or mag_rel <= rel_tol:
            return True, "signed-magnitude"

    return False, "none"


def _numeric_error(value: float, gt: float) -> float:
    errors = [abs(value - gt)]
    if 0.0 <= gt <= 1.0:
        errors.append(abs(value / 100.0 - gt) * 100.0)
        errors.append(abs(value * 100.0 - gt))
    errors.append(abs(abs(value) - abs(gt)))
    return min(errors)


def _allows_signed_magnitude(claim: Dict[str, Any], gt: float) -> bool:
    text = _claim_text(claim).lower()
    if claim.get("snapshot") == "delta":
        negative = (
            "improv", "sooner", "less", "reduced", "decreased", "shorter",
            "faster", "lower", "dropped", "down", "saved", "better",
        )
        positive = (
            "worse", "later", "more", "increased", "longer", "slower",
            "rose", "grew", "higher", "delayed", "extra", "additional",
        )
        if gt < 0:
            return any(tok in text for tok in negative)
        if gt > 0:
            return any(tok in text for tok in positive)
    if gt < 0:
        return any(tok in text for tok in (
            "negative", "over", "behind", "past", "shortfall", "deficit",
        ))
    return False


def _numeric_candidates(evidence: Dict[str, Any], snapshot: Optional[str],
                        metric: str, subject: Optional[str], text: str
                        ) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []

    if metric == "DERIVED: TURNING_INTERVAL":
        for snap in _snapshots_for(snapshot):
            raw = _metric_raw(evidence, snap, metric)
            for label, val in _interval_numeric_candidates(raw):
                out.append((f"{snap}/{metric}/{label}", val))
        return _dedupe_numeric_candidates(out)

    if snapshot == "delta":
        e_t = _metric_raw(evidence, "E_t", metric)
        e_prev = _metric_raw(evidence, "E_t_prev", metric)
        if isinstance(e_t, dict) and isinstance(e_prev, dict):
            keys: List[str]
            if _is_vehicle_id(subject):
                keys = [subject]
            else:
                mentioned = _vehicles_in_text(text)
                keys = mentioned or [k for k, _ in _vehicle_items(e_t)]
            for k in keys:
                a = _to_float(e_t.get(k))
                b = _to_float(e_prev.get(k))
                if a is not None and b is not None:
                    out.append((f"delta/{metric}/{k}", a - b))
        else:
            a = _to_float(e_t)
            b = _to_float(e_prev)
            if a is not None and b is not None:
                out.append((f"delta/{metric}/global", a - b))
        return _dedupe_numeric_candidates(out)

    for snap in _snapshots_for(snapshot):
        raw = _metric_raw(evidence, snap, metric)
        if isinstance(raw, dict):
            if _is_vehicle_id(subject):
                keys = [subject]
            else:
                # If the extractor gave subject=all but the text names a
                # subset ("V1, V2, V3, and V4 all ..."), grade that subset.
                keys = _vehicles_in_text(text) or [k for k, _ in _vehicle_items(raw)]
            for k, v in _vehicle_items(raw, keys):
                f = _to_float(v)
                if f is not None:
                    out.append((f"{snap}/{metric}/{k}", f))
        else:
            f = _to_float(raw)
            if f is not None:
                out.append((f"{snap}/{metric}/global", f))
    return _dedupe_numeric_candidates(out)


def _dedupe_numeric_candidates(cands: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    seen = set()
    out = []
    for label, val in cands:
        key = (label, round(val, 8))
        if key not in seen:
            out.append((label, val))
            seen.add(key)
    return out


def _compatible_metric_family(src: str, dst: str) -> bool:
    if not src or src == "unknown":
        return False
    if src.startswith("P=?") != dst.startswith("P=?"):
        return False
    if src.startswith("DERIVED:"):
        return dst.startswith("DERIVED:")
    families = [
        "pickup_delay", "dropoff_delay", "service_complete",
        "capacity", "carpool", "deadhead", "clear_current_route",
        "dropoff_slack", "vehicle_busy", "bridge", "event",
    ]
    return any(f in src and f in dst for f in families)


def _alternate_numeric_match(claim: Dict[str, Any], evidence: Dict[str, Any],
                             abs_tol: float, rel_tol: float
                             ) -> Optional[Tuple[str, float, str]]:
    metric = _canonical_metric(claim.get("metric"))
    value = _to_float(claim.get("value"))
    if value is None or metric in (None, "unknown", ""):
        return None
    text = _claim_text(claim)
    if _threshold_literal(value, text, str(metric)):
        return None

    best: Optional[Tuple[float, str, float, str]] = None
    metrics = set()
    for snap in ("E_t", "E_t_prev"):
        raw_snap = evidence.get(snap, {})
        if isinstance(raw_snap, dict):
            metrics.update(k for k in raw_snap.keys() if _compatible_metric_family(str(metric), k))

    for alt_metric in metrics:
        if alt_metric == metric:
            continue
        for label, gt in _numeric_candidates(evidence, claim.get("snapshot"),
                                             alt_metric, claim.get("subject"), text):
            ok, mode = _numeric_match(
                value, gt, text, abs_tol, rel_tol,
                allow_signed_magnitude=_allows_signed_magnitude(claim, gt),
            )
            if ok:
                err = _numeric_error(value, gt)
                if best is None or err < best[0]:
                    best = (err, label, gt, mode)
    if best is None:
        return None
    _, label, gt, mode = best
    return label, gt, mode


# ---------- per-kind verification ----------

def verify_numerical(claim: Dict[str, Any], evidence: Dict[str, Any],
                     abs_tol: float = 0.5, rel_tol: float = 0.02
                     ) -> Tuple[str, Any, str]:
    """Numerical claim: V+ if the value matches evidence within tolerance.

    Handles the main extractor failure modes:
      - probability evidence stored as 0..1 but cited as percent,
      - subject=all/unspecified on per-vehicle metrics,
      - nested TURNING_INTERVAL evidence,
      - signed deltas described as positive magnitudes ("10 minutes sooner"),
      - metric-binding slips where the cited value exists under a sibling
        metric with the same subject/snapshot.
    """
    subj = claim.get("subject")
    snapshot = claim.get("snapshot")
    metric = _canonical_metric(claim.get("metric"))
    value = _to_float(claim.get("value"))
    text = _claim_text(claim)

    if value is None:
        return ("U", None, "claim value not numeric")
    if metric in (None, "unknown", ""):
        return ("U", None, "metric unknown")

    if _threshold_literal(value, text, str(metric)):
        return ("U", None, "threshold literal, not an evidence value")

    candidates = _numeric_candidates(evidence, snapshot, metric, subj, text)
    if not candidates:
        alt = _alternate_numeric_match(claim, evidence, abs_tol, rel_tol)
        if alt is not None:
            label, gt, mode = alt
            return ("V+", gt,
                    f"value {value} ≈ alternate {label} {gt:.4f} ({mode}; extractor metric={metric})")
        if snapshot == "delta":
            return ("U", None, "missing snapshot data for delta")
        if snapshot in ("E_t", "E_t_prev"):
            return ("U", None, f"no numeric value for {snapshot}/{metric}/{subj}")
        if snapshot in (None, "unspecified"):
            return ("U", None, "no evidence in either snapshot")
        return ("U", None, f"unknown snapshot {snapshot}")

    # Strict universal claims ("all vehicles have 0 pending requests") should
    # not be satisfied by just one matching vehicle. Range claims are handled
    # by the candidate-level path below.
    if subj == "all" and _looks_universal(text) and not _looks_like_range(text):
        matches = []
        for label, gt in candidates:
            ok, mode = _numeric_match(
                value, gt, text, abs_tol, rel_tol,
                allow_signed_magnitude=_allows_signed_magnitude(claim, gt),
            )
            matches.append((ok, mode, label, gt))
        if matches and all(ok for ok, _, _, _ in matches):
            return ("V+", [gt for _, _, _, gt in matches],
                    f"value {value} matches all {len(matches)} vehicle values")
        # If none match because the number is a threshold, it was already
        # removed above. Otherwise this is a real contradiction of an all-claim.
        if matches and any(ok for ok, _, _, _ in matches):
            bad = [(label, gt) for ok, _, label, gt in matches if not ok][:3]
            return ("V-", [gt for _, _, _, gt in matches],
                    f"value {value} does not match all vehicles; first mismatches={bad}")

    best_miss = None
    for label, gt in candidates:
        ok, mode = _numeric_match(
            value, gt, text, abs_tol, rel_tol,
            allow_signed_magnitude=_allows_signed_magnitude(claim, gt),
        )
        if ok:
            return ("V+", gt, f"value {value} ≈ {label} {gt:.4f} ({mode})")
        err = _numeric_error(value, gt)
        if best_miss is None or err < best_miss[0]:
            best_miss = (err, label, gt)

    alt = _alternate_numeric_match(claim, evidence, abs_tol, rel_tol)
    if alt is not None:
        label, gt, mode = alt
        return ("V+", gt,
                f"value {value} ≈ alternate {label} {gt:.4f} ({mode}; extractor metric={metric})")

    _, label, gt = best_miss
    return ("V-", gt, f"value {value} != any candidate (closest: {label}={gt:.4f})")


_BOOL_FALSE_TOKENS = {
    "no", "none", "false", "clear", "unaffected", "not_affected",
    "not affected", "0", "0.0", "idle", "available", "not busy",
    "no route", "no backlog", "zero",
}
_BOOL_TRUE_TOKENS = {
    "yes", "true", "affected", "bridge", "event", "1", "1.0",
    "busy", "occupied", "full", "tied up",
}
_HIGH_TOKENS = {"high", "high-confidence", "high confidence", "highly confident", "strong"}
_LOW_TOKENS = {"low", "low-confidence", "low confidence", "not confident", "weak"}
_NEAR_ZERO_TOKENS = {"almost no", "near zero", "near-zero", "very low", "unlikely"}
_NEAR_ONE_TOKENS = {"near certainty", "near-certain", "certain", "almost certain", "very likely"}


def _norm_token(v: Any) -> str:
    return re.sub(r"\s+", " ", str(v).strip().lower().replace("_", " "))


def _categorical_candidates(evidence: Dict[str, Any], snapshot: Optional[str],
                            metric: str, subject: Optional[str], text: str
                            ) -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    if metric == "DERIVED: TURNING_INTERVAL":
        for snap in _snapshots_for(snapshot):
            raw = _metric_raw(evidence, snap, metric)
            if raw is not None:
                out.append((f"{snap}/{metric}", raw))
        return out

    for snap in _snapshots_for(snapshot):
        raw = _metric_raw(evidence, snap, metric)
        if isinstance(raw, dict):
            if _is_vehicle_id(subject):
                keys = [subject]
            else:
                keys = _vehicles_in_text(text) or [k for k, _ in _vehicle_items(raw)]
            for k, v in _vehicle_items(raw, keys):
                out.append((f"{snap}/{metric}/{k}", v))
        elif raw is not None:
            out.append((f"{snap}/{metric}/global", raw))
    return out


def _categorical_match(value: Any, gt: Any, metric: str) -> bool:
    claim = _norm_token(value)
    gt_num = _to_float(gt)
    if gt_num is not None:
        if abs(gt_num) <= 1e-9:
            if claim in _BOOL_FALSE_TOKENS:
                return True
            if any(tok in claim for tok in _NEAR_ZERO_TOKENS):
                return True
        if abs(gt_num - 1.0) <= 1e-9:
            if claim in _BOOL_TRUE_TOKENS:
                return True
            if any(tok in claim for tok in _NEAR_ONE_TOKENS):
                return True
        if "STEP_CONFIDENCE" in metric:
            if any(tok in claim for tok in _HIGH_TOKENS) and gt_num >= 0.66:
                return True
            if any(tok in claim for tok in _LOW_TOKENS) and gt_num <= 0.33:
                return True
            if "medium" in claim and 0.33 < gt_num < 0.66:
                return True
        if "TRAFFIC_LEVEL" in metric:
            if "unchanged" in claim or "did not rise" in claim:
                return False
        return False

    gt_norm = _norm_token(gt)
    if claim == gt_norm:
        return True
    if claim in _BOOL_TRUE_TOKENS and gt_norm in _BOOL_TRUE_TOKENS:
        return True
    if claim in _BOOL_FALSE_TOKENS and gt_norm in _BOOL_FALSE_TOKENS:
        return True
    return False


def _categorical_change_match(value: Any, evidence: Dict[str, Any],
                              metric: str, subject: Optional[str]
                              ) -> Optional[Tuple[str, Any, str]]:
    claim = _norm_token(value)
    unchanged = ("unchanged" in claim or "same" in claim or
                 "did not rise" in claim or "didn't rise" in claim)
    increased = ("increased" in claim or "rose" in claim or
                 "higher" in claim or "worse" in claim)
    decreased = ("decreased" in claim or "dropped" in claim or
                 "lower" in claim or "better" in claim)
    if not (unchanged or increased or decreased):
        return None

    e1_raw = _lookup_evidence(evidence, "E_t", metric, subject)
    e0_raw = _lookup_evidence(evidence, "E_t_prev", metric, subject)
    e1 = _to_float(e1_raw)
    e0 = _to_float(e0_raw)
    if e1 is None or e0 is None:
        return None
    delta = e1 - e0
    if unchanged:
        ok = abs(delta) <= 0.5
    elif increased:
        ok = delta > 0.5
    else:
        ok = delta < -0.5
    return ("V+" if ok else "V-", delta,
            f"change claim={value!r}, delta={delta:.4f} (E_t={e1:.4f}, E_t_prev={e0:.4f})")


def verify_categorical(claim: Dict[str, Any], evidence: Dict[str, Any]
                       ) -> Tuple[str, Any, str]:
    """Categorical claim: V+ if str(claim.value) == str(gt) (case-insensitive),
    else V-. Handles "system"/"global" subject by going metric-only when it's
    a scalar metric."""
    subj = claim.get("subject")
    snapshot = claim.get("snapshot")
    metric = _canonical_metric(claim.get("metric"))
    value = claim.get("value")
    text = _claim_text(claim)

    if value is None or (isinstance(value, str) and not value.strip()):
        return ("U", None, "claim value empty")
    if metric in (None, "unknown", ""):
        return ("U", None, "metric unknown")

    change_result = _categorical_change_match(value, evidence, str(metric), subj)
    if change_result is not None:
        return change_result

    if metric == "DERIVED: TURNING_INTERVAL":
        for snap in _snapshots_for(snapshot):
            raw = _metric_raw(evidence, snap, metric)
            if raw is not None and _interval_claim_matches(raw, f"{text} {value}"):
                return ("V+", raw, f"interval claim matches {snap}/{metric}")

    candidates = _categorical_candidates(evidence, snapshot, metric, subj, text)
    if not candidates:
        if snapshot in ("E_t", "E_t_prev"):
            return ("U", None, f"no value for {snapshot}/{metric}/{subj}")
        if snapshot in (None, "unspecified"):
            return ("U", None, "no evidence in either snapshot")
        return ("U", None, f"snapshot {snapshot} not handled for categorical")

    if subj == "all" and _looks_universal(text):
        verdicts = [_categorical_match(value, gt, str(metric)) for _, gt in candidates]
        if verdicts and all(verdicts):
            return ("V+", [gt for _, gt in candidates],
                    f"claim={value!r} matches all {len(candidates)} candidates")
        if verdicts and any(verdicts):
            bad = [(label, gt) for (label, gt), ok in zip(candidates, verdicts) if not ok][:3]
            return ("V-", [gt for _, gt in candidates],
                    f"claim={value!r} does not match all candidates; first mismatches={bad}")

    for label, gt in candidates:
        if _categorical_match(value, gt, str(metric)):
            return ("V+", gt, f"claim={value!r} matches {label}: {gt!r}")

    # For system/global STEP_CONFIDENCE claims, "high confidence" means the
    # selected/top vehicle has high assignment confidence.
    if "STEP_CONFIDENCE" in str(metric) and subj in ("system", "global", "unspecified"):
        nums = [(label, _to_float(gt)) for label, gt in candidates]
        nums = [(label, gt) for label, gt in nums if gt is not None]
        if nums:
            label, gt = max(nums, key=lambda x: x[1])
            if _categorical_match(value, gt, str(metric)):
                return ("V+", gt, f"claim={value!r} matches max {label}: {gt:.4f}")

    label, gt = candidates[0]
    return ("V-", gt, f"claim={value!r}, closest checked={label}: {gt!r}")


def verify_comparative(claim: Dict[str, Any], evidence: Dict[str, Any],
                       assigned_v: Optional[int], closest_v: Optional[int]
                       ) -> Tuple[str, Any, str]:
    """Comparative claim: actual_dir = sign of (subject - comparator) at the
    snapshot, or sign(E_t - E_t_prev) for snapshot=delta. Comparator is the
    (assigned, closest) counterpart. V+ if matches claim.direction, else V-.
    """
    subj = claim.get("subject")
    snapshot = claim.get("snapshot")
    metric = _canonical_metric(claim.get("metric"))
    direction = claim.get("direction")

    if direction not in ("+", "-", "0"):
        return ("U", None, f"invalid direction {direction!r}")
    if metric in (None, "unknown", ""):
        return ("U", None, "metric unknown")

    def _sign(x: float) -> str:
        if abs(x) < 1e-6:
            return "0"
        return "+" if x > 0 else "-"

    def _check_snapshot(snap: str) -> Tuple[str, Any, str]:
        if not (isinstance(subj, str) and subj.startswith("V")):
            # Global/scalar comparative claims usually mean E_t vs E_t_prev.
            e1 = _to_float(_lookup_evidence(evidence, "E_t", metric, subj))
            e0 = _to_float(_lookup_evidence(evidence, "E_t_prev", metric, subj))
            if e1 is not None and e0 is not None:
                actual = _sign(e1 - e0)
                return ("V+" if actual == direction else "V-", actual,
                        f"global delta sign={actual} (E_t={e1:.4f}, E_t_prev={e0:.4f})")
            return ("U", None, "subject not a specific vehicle")

        s_val = _to_float(_lookup_evidence(evidence, snap, metric, subj))
        if s_val is None:
            return ("U", None, f"missing evidence for subject {subj} at {snap}")

        # Pass A: canonical comparator from (assigned, closest)
        other_id = None
        if assigned_v is not None and closest_v is not None and assigned_v != closest_v:
            if subj == f"V{assigned_v}":
                other_id = closest_v
            elif subj == f"V{closest_v}":
                other_id = assigned_v
        if other_id is not None:
            o_val = _to_float(_lookup_evidence(evidence, snap, metric, f"V{other_id}"))
            if o_val is not None:
                actual = _sign(s_val - o_val)
                return ("V+" if actual == direction else "V-", actual,
                        f"sign({subj}-V{other_id})={actual} ({s_val:.4f} vs {o_val:.4f})")

        # Pass B: rank-based fallback. With no canonical comparator, check
        # whether the claim's direction is consistent with subject's rank
        # against ALL other vehicles at this snapshot. "subj < some other"
        # is the truth value of direction "-"; analogous for "+" and "0".
        snap_dict = evidence.get(snap, {}).get(metric)
        if not isinstance(snap_dict, dict):
            return ("U", None, "no per-vehicle metric data for rank fallback")
        others = []
        for k, v in snap_dict.items():
            if k != subj and k.startswith("V"):
                vf = _to_float(v)
                if vf is not None:
                    others.append((k, vf))
        if not others:
            return ("U", None, "no other vehicles to rank against")
        if direction == "-":
            consistent = any(o > s_val for _, o in others)
            extreme_disproof = (s_val == max(s_val, *(o for _, o in others)))
            note = ("subj is max — claim 'less than' is false"
                    if extreme_disproof else "subj < at least one other")
        elif direction == "+":
            consistent = any(o < s_val for _, o in others)
            extreme_disproof = (s_val == min(s_val, *(o for _, o in others)))
            note = ("subj is min — claim 'greater than' is false"
                    if extreme_disproof else "subj > at least one other")
        else:  # "0"
            consistent = all(abs(o - s_val) < 0.5 for _, o in others)
            note = "all values within 0.5"
        return ("V+" if consistent else "V-", direction,
                f"rank-fallback: {note}; subj={s_val:.4f}, "
                f"others={ {k: round(v,2) for k,v in others} }")

    if snapshot == "delta":
        e1 = _to_float(_lookup_evidence(evidence, "E_t", metric, subj))
        e0 = _to_float(_lookup_evidence(evidence, "E_t_prev", metric, subj))
        if e1 is None or e0 is None:
            return ("U", None, "missing snapshot values for delta")
        actual = _sign(e1 - e0)
        return ("V+" if actual == direction else "V-", actual,
                f"delta sign={actual} (E_t={e1:.4f}, E_t_prev={e0:.4f})")

    if snapshot in ("E_t", "E_t_prev"):
        return _check_snapshot(snapshot)

    if snapshot in (None, "unspecified"):
        checked = []
        for snap in ("E_t", "E_t_prev"):
            verdict, gt, reason = _check_snapshot(snap)
            if verdict == "V+":
                return verdict, gt, f"{reason} (snapshot unspecified)"
            if verdict == "V-":
                checked.append((gt, reason))
        if checked:
            gt, reason = checked[0]
            return ("V-", gt, f"no snapshot matched; first check: {reason}")
        return ("U", None, "snapshot unspecified not handled for comparative")

    return ("U", None, f"snapshot {snapshot} not handled for comparative")


def verify_causal(claim: Dict[str, Any], evidence: Dict[str, Any]
                  ) -> Tuple[str, Any, str]:
    return ("N/A", None, "causal claim — not graded")


def verify_claim(claim: Dict[str, Any], evidence: Dict[str, Any],
                 assigned_v: Optional[int], closest_v: Optional[int]
                 ) -> Tuple[str, Any, str]:
    kind = claim.get("kind")
    if kind == "numerical":
        return verify_numerical(claim, evidence)
    if kind == "categorical":
        return verify_categorical(claim, evidence)
    if kind == "comparative":
        return verify_comparative(claim, evidence, assigned_v, closest_v)
    if kind == "causal":
        return verify_causal(claim, evidence)
    return ("U", None, f"unknown kind {kind!r}")


# ---------- aggregation ----------

def aggregate(graded_records: List[Dict[str, Any]],
              id_to_type: Dict[str, int]) -> Dict[str, Any]:
    """Compute V+ / V- / U / N/A rates by kind, type_k, scenario, overall."""
    agg = {
        "by_kind": defaultdict(lambda: Counter()),
        "by_type_k": defaultdict(lambda: Counter()),
        "by_scenario": defaultdict(lambda: Counter()),
        "overall": Counter(),
    }
    for rec in graded_records:
        tk = id_to_type.get(rec["query_id"])
        sc = rec.get("scenario")
        for c in rec.get("graded_claims", []):
            v = c["verdict"]
            kind = c["claim"].get("kind")
            agg["overall"][v] += 1
            agg["by_kind"][kind][v] += 1
            if tk is not None:
                agg["by_type_k"][tk][v] += 1
            if sc is not None:
                agg["by_scenario"][sc][v] += 1

    def _rates(c: Counter) -> Dict[str, Any]:
        graded = c["V+"] + c["V-"] + c["U"]
        return {
            "V+": c["V+"], "V-": c["V-"], "U": c["U"], "N/A": c["N/A"],
            "graded_total": graded,
            "V+_rate": (c["V+"] / graded) if graded else 0.0,
            "V-_rate": (c["V-"] / graded) if graded else 0.0,
            "U_rate":  (c["U"]  / graded) if graded else 0.0,
        }

    return {
        "overall": _rates(agg["overall"]),
        "by_kind": {k: _rates(v) for k, v in sorted(agg["by_kind"].items())},
        "by_type_k": {str(k): _rates(v) for k, v in sorted(agg["by_type_k"].items())},
        "by_scenario": {str(k): _rates(v) for k, v in sorted(agg["by_scenario"].items())},
    }


# ---------- pipeline ----------

def grade_record(claims_rec: Dict[str, Any],
                 evidence_rec: Dict[str, Any]) -> Dict[str, Any]:
    """Grade all claims in one record. Returns the per-record graded output."""
    qid = claims_rec["query_id"]
    evidence = evidence_rec.get("evidence", {}) or {}

    # assigned/closest already stored in extraction_meta by claim_extractor
    em = claims_rec.get("extraction_meta", {}) or {}
    assigned_v = em.get("assigned_v")
    closest_v = em.get("closest_v")

    graded = []
    for claim in claims_rec.get("claims", []):
        verdict, gt, reason = verify_claim(claim, evidence, assigned_v, closest_v)
        graded.append({
            "claim": claim,
            "verdict": verdict,        # "V+" | "V-" | "U" | "N/A"
            "gt_value": gt,
            "reason": reason,
        })

    return {
        "query_id": qid,
        "scenario": evidence_rec.get("scenario"),
        "type_k": evidence_rec.get("type_k"),
        "assigned_v": assigned_v,
        "closest_v": closest_v,
        "n_claims": len(graded),
        "verdict_counts": dict(Counter(g["verdict"] for g in graded)),
        "graded_claims": graded,
    }


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--claims", default=DEFAULT_CLAIMS, help="claims_240.json")
    p.add_argument("--evidence", default=DEFAULT_EVIDENCE, help="evidence_240.json")
    p.add_argument("--output", default=DEFAULT_OUTPUT,
                   help="output graded JSON")
    p.add_argument("--query-csv", default=DEFAULT_QUERY_CSV,
                   help="core_queries_240.csv (for type_k mapping)")
    p.add_argument("--query-ids", nargs="+",
                   help="grade only these query_ids (default: all)")
    p.add_argument("--abs-tol", type=float, default=0.5,
                   help="absolute tolerance for numerical match (default 0.5)")
    p.add_argument("--rel-tol", type=float, default=0.02,
                   help="relative tolerance for numerical match (default 0.02 = 2%%)")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.claims) as f:
        claims_recs = json.load(f)
    with open(args.evidence) as f:
        evidence_recs = json.load(f)

    evidence_by_id = {r["query_id"]: r for r in evidence_recs}
    id_to_type: Dict[str, int] = {}
    if os.path.exists(args.query_csv):
        with open(args.query_csv) as f:
            for row in csv.DictReader(f):
                id_to_type[row["query_id"]] = int(row["type_k"])

    if args.query_ids:
        q_set = set(args.query_ids)
        claims_recs = [r for r in claims_recs if r["query_id"] in q_set]

    # apply tolerance overrides through closure on verify_numerical
    if args.abs_tol != 0.5 or args.rel_tol != 0.02:
        global verify_numerical
        _orig = verify_numerical
        def verify_numerical(claim, evidence):
            return _orig(claim, evidence,
                         abs_tol=args.abs_tol, rel_tol=args.rel_tol)

    graded_records: List[Dict[str, Any]] = []
    for rec in claims_recs:
        qid = rec["query_id"]
        ev = evidence_by_id.get(qid)
        if ev is None:
            print(f"[warn] no evidence for {qid}, skipping")
            continue
        graded_records.append(grade_record(rec, ev))

    summary = aggregate(graded_records, id_to_type)

    out = {
        "summary": summary,
        "records": graded_records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # Print summary
    print(f"\nGraded {len(graded_records)} records, "
          f"{sum(r['n_claims'] for r in graded_records)} claims total.\n")

    def _print_block(label, d):
        v = d["V+"]; minus = d["V-"]; u = d["U"]; na = d["N/A"]; g = d["graded_total"]
        if g == 0:
            print(f"  {label:25s}: (no graded claims)")
            return
        print(f"  {label:25s}: V+={v:>4} ({d['V+_rate']*100:>5.1f}%) "
              f"V-={minus:>4} ({d['V-_rate']*100:>5.1f}%) "
              f"U={u:>4} ({d['U_rate']*100:>5.1f}%) "
              f"N/A={na:>4} | total graded={g}")

    print("=== Overall ===")
    _print_block("overall", summary["overall"])

    print("\n=== By kind ===")
    for k, d in summary["by_kind"].items():
        _print_block(str(k), d)

    print("\n=== By type_k ===")
    for k, d in summary["by_type_k"].items():
        _print_block(f"type {k}", d)

    print("\n=== By scenario ===")
    for k, d in summary["by_scenario"].items():
        _print_block(f"scenario {k}", d)

    print(f"\nGraded output -> {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
