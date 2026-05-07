# Evaluation Status Summary

Last updated: 2026-04-30

This file summarizes the current evaluation status for the NS-XAI paper. The
main automated evaluation package is largely complete. The main remaining
paper-critical item is the user study.

## Overall Status

| Area | Status | Notes |
|---|---|---|
| 240-query benchmark setup | Done | Same query prompts are used across query classification, explanation generation, faithfulness, baselines, ablations, and Delta-Recovery. |
| Query classification | Done | 240 original prompts. |
| Paraphrase classification stability | Done | 240 base prompts x 5 phrasings = 1200 phrasings. |
| Numerical faithfulness | Done | Full method, B1, B2, B4, A2, A3, A4. |
| Delta-Recovery | Done | Full method, B1, B2, B4, A2, A3, A4. |
| Baselines | Done | B1 trajectory-only, B2 trajectory + policy, B4/A1 single-snapshot. |
| Core ablations | Done | A1/B4, A2, A3, A4. |
| Ours paraphrase robustness | Done | Explanation-level robustness diagnostic for the full method. |
| Compactness + latency | Done | Summary generated from existing JSON/log outputs. |
| Per-type/per-scenario breakdowns | Done | Appendix-ready CSV/JSON tables generated. |
| User study | In progress | Main remaining paper-critical evaluation block. |
| Optional appendix diagnostics | Not run | B3, Adebayo sanity check, sufficiency/comprehensiveness, BERTScore, LLM robustness. |

## Benchmark And Query Reliability

### Same 240-query benchmark

Status: Done.

Core files:

- `evaluation/core_queries_240.csv`
- `evaluation/queries.txt`
- `evaluation/labels.txt`
- `evaluation/faithfulness/evidence_240.json`

Current wording for the paper:

> We evaluate all automated components on the same 240-query benchmark. Query
> classification, explanation generation, faithfulness, baselines, ablations,
> and Delta-Recovery are evaluated on the original 240 prompts. For paraphrase
> robustness, we use these same 240 prompts as base queries and generate four
> paraphrases per query, keeping evidence fixed.

Legacy query-classification files were moved to:

- `evaluation/query_reliability/legacy/`

### Query classification on original 240 prompts

Status: Done.

What it evaluates:

- Whether the natural-language query is mapped to the intended query type.
- Uses exactly the same 240 prompts as the explanation benchmark.

Headline result:

| Metric | Value |
|---|---:|
| Accuracy | 98.33% |
| Macro precision | 99.22% |
| Macro recall | 98.33% |
| Macro F1 | 98.74% |

Outputs:

- `evaluation/classification_results.csv`
- `evaluation/query_reliability/current_classification.log`

### Paraphrase classification stability

Status: Done.

What it evaluates:

- Each of the 240 base prompts is expanded into 4 paraphrases plus the original
  wording.
- Measures whether query classification remains stable under wording changes.

Headline result:

| Metric | Value |
|---|---:|
| Phrasings | 1200 |
| Accuracy | 97.33% |
| Macro precision | 98.28% |
| Macro recall | 97.33% |
| Macro F1 | 97.74% |

Outputs:

- `evaluation/query_reliability/paraphrase_queries.txt`
- `evaluation/query_reliability/paraphrase_labels.txt`
- `evaluation/query_reliability/paraphrase_classification_results.csv`
- `evaluation/query_reliability/paraphrase_classification.log`

## Faithfulness: Numerical Fidelity

Status: Done.

What it evaluates:

- Explanations are decomposed into atomic claims.
- Numerical claims are checked against the evidence available to each method.
- Verdicts:
  - `V+`: supported by evidence.
  - `V-`: contradicted by evidence.
  - `U`: unverifiable against available evidence.
  - `N/A`: not graded for that verifier category.

Primary result should use numerical claims.

### Numerical-claim results

| Method | V+ | V- | U | Graded numerical claims |
|---|---:|---:|---:|---:|
| Full NS-XAI | 92.38% | 3.30% | 4.32% | 1574 |
| B1 trajectory-only | 88.17% | 0.06% | 11.78% | 1690 |
| B2 trajectory + policy | 83.55% | 4.33% | 12.12% | 1246 |
| B4/A1 single-snapshot | 93.54% | 0.35% | 6.11% | 573 |
| A2 PCTL only | 97.67% | 0.66% | 1.66% | 602 |
| A3 derived only | 95.88% | 2.70% | 1.42% | 777 |
| A4 fixed reference | 94.70% | 2.78% | 2.52% | 1151 |

All-claim rates are available in `evaluation/tables/faithfulness_overall.csv`.

Main output files:

- Full method:
  - `evaluation/faithfulness/evidence_240.json`
  - `evaluation/faithfulness/claims_240.json`
  - `evaluation/faithfulness/graded_claims_240.json`
- B1:
  - `evaluation/baselines/b1_trajectory_only/evidence_240.json`
  - `evaluation/baselines/b1_trajectory_only/claims_240.json`
  - `evaluation/baselines/b1_trajectory_only/graded_claims_240.json`
- B2:
  - `evaluation/baselines/b2_trajectory_policy/evidence_240.json`
  - `evaluation/baselines/b2_trajectory_policy/claims_240.json`
  - `evaluation/baselines/b2_trajectory_policy/graded_claims_240.json`
- B4/A1:
  - `evaluation/baselines/b4_single_snapshot/evidence_240.json`
  - `evaluation/baselines/b4_single_snapshot/claims_240.json`
  - `evaluation/baselines/b4_single_snapshot/graded_claims_240.json`
- A2:
  - `evaluation/ablations/a2_pctl_only/evidence_240.json`
  - `evaluation/ablations/a2_pctl_only/claims_240.json`
  - `evaluation/ablations/a2_pctl_only/graded_claims_240.json`
- A3:
  - `evaluation/ablations/a3_derived_only/evidence_240.json`
  - `evaluation/ablations/a3_derived_only/claims_240.json`
  - `evaluation/ablations/a3_derived_only/graded_claims_240.json`
- A4:
  - `evaluation/ablations/a4_fixed_reference/evidence_240.json`
  - `evaluation/ablations/a4_fixed_reference/claims_240.json`
  - `evaluation/ablations/a4_fixed_reference/graded_claims_240.json`

## Temporal Faithfulness: Delta-Recovery

Status: Done.

What it evaluates:

- An independent verifier reads the explanation and tries to recover the
  direction and magnitude of cross-snapshot changes.
- Primary metric: sign-match over all target delta components.
- Supporting metrics: delta coverage, sign-match among recovered components,
  magnitude MAE, Spearman correlation, scalar interval hit rate.

Headline results:

| Method | Sign-match all | Delta coverage | Sign-match recovered | Magnitude MAE | Spearman | Scalar hit |
|---|---:|---:|---:|---:|---:|---:|
| Full NS-XAI | 34.22% | 36.49% | 93.77% | 15.77 | 0.902 | 100.00% |
| B1 trajectory-only | 0.97% | 2.21% | 44.12% | 32.43 | -0.279 | 0.00% |
| B2 trajectory + policy | 1.10% | 3.70% | 29.82% | 26.80 | -0.431 | 0.00% |
| B4/A1 single-snapshot | 6.10% | 13.05% | 46.77% | 44.55 | -0.051 | 0.00% |
| A2 PCTL only | 10.00% | 14.68% | 68.14% | 21.85 | 0.644 | 0.00% |
| A3 derived only | 13.38% | 17.40% | 76.87% | 21.90 | 0.721 | 100.00% |
| A4 fixed reference | 15.39% | 24.55% | 62.70% | 48.78 | 0.402 | 76.67% |

Outputs:

- `evaluation/delta_recovery/ours_graded_240.json`
- `evaluation/delta_recovery/b1_graded_240.json`
- `evaluation/delta_recovery/b2_graded_240.json`
- `evaluation/delta_recovery/b4_graded_240.json`
- `evaluation/delta_recovery/a2_graded_240.json`
- `evaluation/delta_recovery/a3_graded_240.json`
- `evaluation/delta_recovery/a4_graded_240.json`
- Corresponding prediction files are in the same directory.

## Baselines

Status: Done for B1, B2, B4/A1.

### B1: trajectory-only post-hoc

What it does:

- LLM receives executed trajectory information only.
- No PCTL analysis.
- No cross-snapshot evidence.

Outputs:

- `evaluation/baselines/b1_trajectory_only/`

### B2: trajectory + policy post-hoc

What it does:

- LLM receives trajectory plus planner policy/search statistics.
- No PCTL analysis.
- No cross-snapshot evidence.

Outputs:

- `evaluation/baselines/b2_trajectory_policy/`

### B4/A1: single-snapshot

What it does:

- Uses only current-snapshot evidence.
- Removes previous/reference snapshot values, temporal deltas, recovery
  intervals, and temporal algorithm-state fields.

Outputs:

- `evaluation/baselines/b4_single_snapshot/`

### B3: confident-wrong adversarial

Status: Not run.

Rationale:

- Optional stress-test baseline.
- Not necessary for the current main evaluation package.

## Ablations

Status: Done.

### A1/B4: single-snapshot

Same as B4 above.

### A2: PCTL only

What it does:

- Keeps PCTL Analysis Results.
- Removes Algorithm State and Derived Metrics Summary.

Outputs:

- `evaluation/ablations/a2_pctl_only/`

### A3: derived metrics only

What it does:

- Keeps Algorithm State and Derived Metrics Summary.
- Removes PCTL Analysis Results.

Outputs:

- `evaluation/ablations/a3_derived_only/`

### A4: fixed reference snapshot

What it does:

- Compares current evidence against a fixed initial reference snapshot rather
  than the adaptive previous/reference snapshot.
- MDP_tn cache is redirected under the evaluation directory.

Outputs:

- `evaluation/ablations/a4_fixed_reference/`
- `evaluation/ablations/a4_fixed_reference/mdp_tn_cache/`

## Robustness

### Ours paraphrase robustness

Status: Done.

What it evaluates:

- For each of 240 base prompts, generate 4 paraphrases while keeping evidence
  fixed.
- Compare the explanations within each 5-phrasing group.
- Metrics:
  - numerical-value Jaccard
  - claim-set Jaccard

Headline result:

| Metric | Value |
|---|---:|
| Base query groups | 240 |
| Pairwise comparisons | 2400 |
| Numerical-value Jaccard mean | 0.644 |
| Claim-set Jaccard mean | 0.216 |

Outputs:

- `evaluation/robustness/paraphrase/ours_evidence.json`
- `evaluation/robustness/paraphrase/ours_claims.json`
- `evaluation/robustness/paraphrase/ours_summary.json`
- `evaluation/robustness/paraphrase/ours_generate.log`
- `evaluation/robustness/paraphrase/ours_claim_extractor.log`

Important interpretation:

- This is an ours-only robustness diagnostic.
- Do not claim "ours is more robust than baselines" unless B1/B2/B4
  paraphrase robustness is also run.

### Baseline paraphrase robustness

Status: Not run.

Current plan:

- Time-permitting appendix follow-up.
- If only one baseline is added, prioritize B4 or B2.

### BERTScore

Status: Not run.

Rationale:

- Secondary NLG-style robustness metric.
- Not treated as a faithfulness proxy.
- Not currently needed.

## Compactness And Latency

Status: Done.

What it evaluates:

- Explanation word count.
- Atomic claim count.
- Unique numerical values cited.
- Observed record-level generation latency from each method's `elapsed_sec`.

Headline result:

| Method | Mean words | Mean claims | Mean unique nums | Mean latency | Median latency |
|---|---:|---:|---:|---:|---:|
| Full NS-XAI | 197.2 | 9.7 | 5.6 | 23.21s | 18.16s |
| B1 trajectory-only | 138.2 | 9.3 | 5.8 | 2.20s | 2.01s |
| B2 trajectory + policy | 153.8 | 7.3 | 5.0 | 2.34s | 2.09s |
| B4/A1 single-snapshot | 148.4 | 5.5 | 2.3 | 2.52s | 2.35s |
| A2 PCTL only | 122.9 | 4.5 | 1.7 | 2.21s | 2.04s |
| A3 derived only | 142.9 | 6.1 | 3.1 | 2.27s | 2.24s |
| A4 fixed reference | 190.6 | 6.8 | 4.2 | 27.47s | 2.51s |

Notes:

- `elapsed_sec` is observed record-level wall-clock time, not a clean
  hardware-normalized stage profile.
- A4 mean latency is skewed by fixed-reference rebuild outliers; median is more
  representative for a compact paper statement.

Outputs:

- `evaluation/efficiency/compactness_latency_summary.py`
- `evaluation/efficiency/compactness_latency_summary.json`
- `evaluation/efficiency/compactness_latency_by_method.csv`
- `evaluation/efficiency/compactness_latency_per_record.csv`

## Per-Type And Per-Scenario Breakdown Tables

Status: Done.

What it provides:

- Appendix-ready CSV tables for faithfulness and Delta-Recovery.
- Includes overall, per-kind, per-query-type, per-scenario, and combined
  type/kind or scenario/kind breakdowns where applicable.

Outputs:

- `evaluation/tables/generate_breakdown_tables.py`
- `evaluation/tables/breakdown_tables.json`
- `evaluation/tables/faithfulness_overall.csv`
- `evaluation/tables/faithfulness_by_kind.csv`
- `evaluation/tables/faithfulness_by_type.csv`
- `evaluation/tables/faithfulness_by_scenario.csv`
- `evaluation/tables/faithfulness_by_type_kind.csv`
- `evaluation/tables/faithfulness_by_scenario_kind.csv`
- `evaluation/tables/delta_recovery_overall.csv`
- `evaluation/tables/delta_recovery_by_type.csv`
- `evaluation/tables/delta_recovery_by_scenario.csv`

Validation checks:

- Full-method numerical faithfulness in the generated tables matches the
  expected 92.38% V+ rate.
- Full-method Delta-Recovery in the generated tables matches the expected
  34.22% sign-match(all), 36.49% coverage, and 0.902 Spearman.

## User Study

Status: In progress.

Planned role in paper:

- Main-paper plausibility/usefulness evaluation.
- Should be clearly separated from faithfulness.
- Recommended reported measures:
  - understandability
  - non-stationarity awareness
  - calibrated trust / decision prediction if available

Not yet summarized in evaluation outputs.

## Optional Diagnostics Not Run

These are not required for the current automated evaluation package.

| Diagnostic | Status | Current recommendation |
|---|---|---|
| B3 confident-wrong adversarial baseline | Not run | Optional stress test. |
| Adebayo-style sanity check | Not run | Appendix-only if time allows. |
| Sufficiency/comprehensiveness | Not run | Skip for now; lower cost-benefit and can look like LLM judging LLM. |
| BERTScore | Not run | Secondary NLG metric; not necessary. |
| B1/B2/B4 paraphrase robustness | Not run | Only needed for comparative robustness claims. |
| LLM choice robustness | Not run | Optional appendix if reviewers are likely to ask. |

## Recommended Next Steps

1. Build main-paper result tables from:
   - `evaluation/tables/faithfulness_by_kind.csv`
   - `evaluation/tables/delta_recovery_overall.csv`
   - `evaluation/query_reliability/current_classification.log`
   - `evaluation/query_reliability/paraphrase_classification.log`
   - `evaluation/robustness/paraphrase/ours_summary.json`
   - `evaluation/efficiency/compactness_latency_by_method.csv`
2. Finish and summarize the user study.
3. Use appendix tables from `evaluation/tables/` for full breakdowns.
4. Avoid claiming comparative paraphrase robustness unless baseline
   paraphrase robustness is run.
