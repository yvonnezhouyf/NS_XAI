# Faithfulness Evaluation Status

Last updated: 2026-04-29

## Scope

This folder contains the completed full-run artifacts for **Evaluation Plan
Dimension 1A: Numerical Fidelity** on the 240 core NS-XAI paratransit queries.

Dimension 1 in `evaluation/Evaluation_Plan.md` has three parts:

- **1A Numerical Fidelity**: completed here.
- **1B Sufficiency / Comprehensiveness**: not implemented yet.
- **1C Sanity Check**: not implemented yet.

So Dimension 1 as a whole is not fully complete, but the faithfulness claim
extraction and rule-based verification pipeline for 1A is complete for the
current full 240-query run.

## Final Files

Keep these files in this directory:

- `generate_evidence.py`: stage 0, generates explanations and evidence.
- `evidence_240.json`: final full evidence and explanations for 240 queries.
- `claim_extractor.py`: stage 1, extracts atomic claims from explanations.
- `claims_240.json`: final full extracted claims for 240 queries.
- `claim_verifier.py`: stage 2, rule-based claim verifier.
- `graded_claims_240.json`: final full graded claim output and summary.
- `STATUS.md`: this status and reproduction note.

Pilot files, old claim files, logs, and Python cache files were removed from
this folder to leave only the final full-run artifacts and scripts.

## Current Full Results

Command:

```bash
cd .
python evaluation/faithfulness/claim_verifier.py \
  --output evaluation/faithfulness/graded_claims_240.json
```

Summary from `graded_claims_240.json`:

```text
Graded 240 records, 2339 claims total.

Overall:
  V+ = 1626 (73.3%)
  V- =  202 ( 9.1%)
  U  =  391 (17.6%)
  N/A = 120

By kind:
  numerical    V+ = 1454 (92.4%), V- =  52 ( 3.3%), U =  68 ( 4.3%)
  categorical  V+ =  124 (25.3%), V- = 120 (24.5%), U = 246 (50.2%)
  comparative  V+ =   48 (31.0%), V- =  30 (19.4%), U =  77 (49.7%)
  causal       N/A = 120
```

For Dimension 1A reporting, the most important number is the **numerical**
row, because the method in the evaluation plan is explicitly numerical
fidelity. The overall row is useful context, but it mixes numerical claims with
categorical, comparative, and causal claims.

## What The Verifier Handles

`claim_verifier.py` is still rule-based. It now handles:

- evidence probabilities stored as `0..1` but cited as percentages,
- `DERIVED: STEP_CONFIDENCE` percentage citations,
- `subject=all` and `subject=unspecified` for per-vehicle metrics,
- nested `DERIVED: TURNING_INTERVAL` evidence,
- signed deltas stated as positive magnitudes, e.g. "10 minutes sooner",
- common categorical boolean forms such as `none`, `clear`, `near certainty`,
- conservative alternate-metric recovery when the cited value exists under a
  sibling metric with the same subject/snapshot.

Threshold literals such as `60+ minute delay` and `15 minutes or less` are
treated as unverifiable thresholds, not evidence values.

## Remaining Limitations

Most remaining `U` claims are outside the strict numerical evidence surface:

- `metric=unknown` qualitative claims,
- algorithm-state statements such as "stable phase" or "high confidence",
- comparative claims without a concrete vehicle subject,
- categorical claims whose semantics are too broad for a rule-only verifier.

These should not be interpreted as numerical hallucinations. For paper
reporting, separate the main numerical-fidelity result from the broader
all-claim summary.

## Reproduction

Full 1A pipeline:

```bash
cd .

# Stage 0: already completed; re-run only if evidence/explanations change.
python evaluation/faithfulness/generate_evidence.py \
  --output evaluation/faithfulness/evidence_240.json \
  --resume

# Stage 1: LLM call, costs API usage. Re-run only if extractor prompt changes.
python evaluation/faithfulness/claim_extractor.py \
  --output evaluation/faithfulness/claims_240.json \
  --workers 8

# Stage 2: local rule-based verification. Safe to re-run anytime.
python evaluation/faithfulness/claim_verifier.py \
  --output evaluation/faithfulness/graded_claims_240.json
```

