# Evaluation Plan

## Paper Placement Summary

Use the evaluation section as a compact **main-paper evidence chain**, with
details and secondary diagnostics moved to the appendix.

### Main Paper

1. **Experimental setup and benchmark**: paratransit domain, snapshots, 240
   core queries, query types, scenarios, baselines, and implementation summary.
2. **Query/specification reliability**: query classification accuracy and
   paraphrase stability of the generated formal specification.
3. **Faithfulness**: numerical fidelity as the primary faithfulness result.
   Report V+, V−, and U rates for numerical claims, comparing ours against the
   main baselines.
4. **Temporal faithfulness**: Δ-Recovery as the central non-stationary result.
   Report sign-match accuracy, magnitude MAE, and Spearman correlation.
5. **Robustness and ablations**: paraphrase robustness plus the core ablations
   needed to justify the framework components: no cross-snapshot, PCTL only,
   derived metrics only, and adaptive vs fixed reference snapshot.
6. **Plausibility user study**: concise main-paper table with understandability,
   non-stationarity awareness, and calibrated trust / decision prediction if
   available.

### Appendix

1. Full Co-12 mapping and metric definitions.
2. Full claim-extraction and verification protocol, including all-claim
   summaries broken down by numerical, categorical, comparative, and causal
   claims.
3. Sufficiency / comprehensiveness reconstruction diagnostics.
4. Adebayo-style sanity check details and examples.
5. Full robustness tables, BERTScore, compactness, and latency breakdowns.
6. Full ablation tables and per-query / per-scenario results.
7. User-study protocol, screenshots, instructions, demographics, IRB statement,
   statistical tests, reliability analysis, and free-response coding if used.
8. Prompts, templates, hyperparameters, hardware, random seeds, and
   reproducibility details.

Recommended main-paper ordering:

| Main Section | Contents | Placement |
|---|---|---|
| 5.1 Setup | Domain, snapshots, query benchmark, baselines | Main |
| 5.2 Query Reliability | Classification accuracy, spec stability | Main |
| 5.3 Faithfulness | Numerical fidelity table | Main |
| 5.4 Temporal Faithfulness | Δ-Recovery table | Main |
| 5.5 Robustness and Ablations | Paraphrases + A1/A2/A3/A4 compact table | Main |
| 5.6 User Study | Plausibility and calibrated trust | Main |
| 5.7 Efficiency | One-sentence summary; full table in appendix | Main short / Appendix full |

---

## Organizing Principle

Evaluation organized along six dimensions of the **Co-12 framework** (Nauta et al., ACM Comp Surveys 2023), with all primary metrics being **functionally-grounded** (Doshi-Velez & Kim, 2017) — i.e., automated, no human required. Plausibility is separated from faithfulness per **Jacovi & Goldberg (ACL 2020)**.

| Dimension (Co-12) | Method | Auto? |
|---|---|---|
| 1. Correctness / Faithfulness | Numerical fidelity + ERASER-style sufficiency/comprehensiveness + Adebayo sanity check | ✓ |
| 2. Temporal Faithfulness (proposed) | Δ-Recovery test | ✓ |
| 3. Contrastivity | 8-type contrastive question coverage (Krarup et al. JAIR 2021 framing) | ✓ |
| 4. Robustness | Numerical consistency + Claim-set Jaccard | ✓ |
| 5. Compactness | Word count, claim count, unique cited values | ✓ |
| 6. Plausibility | User study (separated from faithfulness, per Jacovi & Goldberg 2020) | human |

**Placement note**: use this six-dimension table as a compact organizing
summary in the appendix or as a short paragraph in the main paper. The main
paper should be organized by the evidence chain above rather than by all six
Co-12 dimensions, because NeurIPS reviewers will look first for supported
claims, baselines, ablations, and reproducibility.

---

## Baselines — Main Paper

All comparisons run on the same query benchmark.

- **B1 — Trajectory-only post-hoc**: LLM given executed trajectory only.
- **B2 — Trajectory + policy post-hoc**: LLM given trajectory + planner's Q-values / action probabilities (comparable info to ours, minus PCTL evaluation).
- **B3 — Confident-wrong adversarial**: LLM prompted to generate a fluent, authoritative explanation citing PCTL-style probability values, but with **no actual PCTL evaluation** — values are fabricated. Surface-form indistinguishable from ours.
- **B4 — Single-snapshot ours**: framework restricted to current snapshot only (= Ablation 1, dual-purpose).

**Main-paper use**:
- Treat B1, B2, and B4 as the primary baselines.
- Treat B3 as a negative-control / stress-test baseline rather than the main
  competitor.
- For B4/A1, use a single-snapshot prompt variant derived from the full
  explanation prompt, and construct the LLM input from current-snapshot
  evidence only. Current PCTL values and current derived metrics may remain;
  previous/reference-snapshot values, temporal deltas, recovery intervals, and
  temporal algorithm-state fields must be absent from the input, not merely
  discouraged by instruction.
- If space permits, include an adapted existing formal/planning explanation
  baseline in the main table; otherwise put it in the appendix with a clear
  note that it is not designed for cross-snapshot non-stationarity.

---

## Dimension 0 — Query and Specification Reliability — Main Paper

Before evaluating explanations, report that the natural-language interface
reliably maps user queries to the intended formal specification.

**Reported in main paper**:
- Query classification accuracy, macro-F1, and confusion matrix summary.
- Paraphrase stability: fraction of paraphrases that map to the same query type
  and same formal specification.
- Brief failure analysis for misclassified or unsupported queries.

**Appendix**:
- Full confusion matrix.
- All prompt templates and label definitions.
- Per-type classification precision / recall / F1.

---

## Dimension 1 — Correctness / Faithfulness — Main Paper + Appendix

### 1A. Numerical Fidelity

For each generated explanation, decompose into atomic numerical claims (LLM-based). Each claim classified as:
- **V+**: matches ground-truth PCTL value (within tolerance ε)
- **V−**: contradicts ground-truth PCTL value
- **U**: unverifiable against the method's available evidence

For B1/B2/B3, "evidence" = whatever input the baseline received (trajectory, Q-values, or LLM's own context). Fair to all methods: a baseline can correctly cite a number from its input without penalty; only fabricated numbers count as unverifiable.

**Reported**: V+ rate (numerical fidelity), V− rate (hallucination), U rate (unverifiability).

**Main paper**:
- Primary faithfulness table.
- Report the numerical-claim row as the headline result.
- Compare ours against B1, B2, B4, and optionally B3 as a negative control.

**Appendix**:
- Full all-claim summary, including categorical, comparative, and causal claims.
- Per-query-type and per-scenario breakdowns.
- Claim-extraction prompt, verifier rules, tolerance ε, and examples of V+,
  V−, U, and N/A.

### 1B. Sufficiency / Comprehensiveness (DeYoung et al. ERASER, ACL 2020)

Adapted from ERASER's standard faithfulness operationalization:

- **Sufficiency**: feed only the explanation's content to a downstream LLM; ask it to predict the cross-snapshot shift direction (which components of Δ(q) are positive/negative). Compare to ground-truth Δ(q). High accuracy ⇒ explanation is sufficient.
- **Comprehensiveness**: from full evidence (E_t, E_t', Δ(q)), remove the components referenced by the explanation; feed remainder to LLM and predict shift. Large degradation ⇒ explanation comprehensively captures the relevant evidence.

**Placement**: appendix diagnostic. Do not make this the main faithfulness
claim unless the protocol is converted into a tightly structured reconstruction
task with JSON outputs and deterministic scoring. This reduces the risk that
reviewers see it as an LLM judging another LLM.

### 1C. Sanity Check (Adebayo et al., NeurIPS 2018)

Randomly shuffle the numerical values in (E_t, E_t', Δ(q)) before feeding to the explanation generator. A faithful framework should produce visibly different explanations under randomized evidence. Report:
- **Explanation divergence rate**: fraction of randomized inputs where the resulting explanation differs from the original (claim-set Jaccard < 0.5).

A high divergence rate confirms the framework is genuinely conditioning on PCTL evidence rather than producing template-driven text.

**Placement**: appendix, with a one-sentence main-paper mention only if the
result is strong and space allows.

---

## Dimension 2 — Temporal Faithfulness (Δ-Recovery) — Main Paper

Cross-snapshot fidelity has no widely-adopted metric in existing XAI surveys. We propose **Δ-Recovery** as a functionally-grounded test of whether the explanation correctly communicates the cross-snapshot shift.

**Protocol**:
- Independent LLM (different from explanation generator) reads explanation
- For each component of Spec(q), LLM predicts Δ̂_i = sign and (optionally) magnitude
- Compare against ground-truth Δ_i

**Metrics**:
- **Sign-match accuracy**: fraction of components where Δ̂_i and Δ_i agree in sign
- **Magnitude MAE** (where applicable): mean absolute error between predicted and ground-truth magnitude
- **Spearman correlation** between Δ̂ and Δ across components

This metric directly probes the unique non-stationary contribution: a fluent explanation citing wrong numbers will recover Δ̂ that does not correlate with ground truth.

**Framing in paper**: do **not** claim a specific survey identified this as a gap. Instead, argue from your own related-work review that no existing XAI evaluation specifically targets cross-snapshot fidelity for sequential decision-making.

**Main paper**:
- This is the central non-stationary evaluation.
- Compare ours against B1, B2, and B4.
- Report sign-match accuracy as the primary number; use magnitude MAE and
  Spearman correlation as supporting metrics.

**Appendix**:
- Per-query-type, per-scenario, and per-component Δ-Recovery results.
- Parser / evaluator prompt and structured output schema.

---

## Dimension 3 — Contrastivity — Main Paper Setup / Appendix Details

Krarup et al. (JAIR 2021) showed plan-related user questions are predominantly contrastive ("why A rather than B?"). The 8 core query types in your benchmark already map to a contrastive question taxonomy.

**Reported**: Coverage = number of distinct contrastive question types the framework supports as Spec(q) constructions. Reframing of existing result; no new experiment needed.

**Placement**:
- Main paper: describe this in the benchmark/setup section, not as a standalone
  result.
- Appendix: include the full query taxonomy and mapping from each query type to
  atomic propositions, temporal operators, derived metrics, and example queries.

---

## Dimension 4 — Robustness — Main Paper + Appendix

For each base query, generate explanations from all 5 paraphrases holding (S_t, S_t') fixed.

**Primary metrics**:
- **Numerical consistency**: pairwise Jaccard over the sets of numerical values cited (within tolerance ε)
- **Claim-set Jaccard**: pairwise Jaccard over the decomposed atomic claim sets

**Secondary metric**:
- **BERTScore (F1)**: pairwise across explanations. Reported for comparability with prior NLG-style work, **not** treated as a faithfulness proxy (BERTScore is insensitive to numerical substitutions).

Current execution plan: run paraphrase robustness for the full method first as
a robustness diagnostic over the same 240 base queries. Do not make comparative
"more robust than baselines" claims unless baseline robustness is also run.
If time allows, add B4 or B2 as the most relevant comparison; B1/B2/B4 full
robustness can be left as appendix follow-up because it requires substantially
more explanation-generation calls.

**Main paper**:
- Include a compact robustness table for paraphrase stability.
- Primary emphasis should be numerical consistency and claim-set Jaccard.

**Appendix**:
- Full pairwise tables.
- BERTScore results.
- Qualitative examples where lexical wording changes but evidence remains
  stable.

---

## Dimension 5 — Compactness — Appendix

**Reported per method**:
- Word count per explanation
- Atomic claim count per explanation
- Unique numerical values cited per explanation

Purpose: rule out the alternative explanation that our framework wins by being verbose. If our explanations are **shorter** or **comparable in length** to baselines while scoring higher on faithfulness, the contribution is strengthened.

**Placement**: appendix, unless reviewers are likely to suspect that gains come
from much longer explanations. In the main paper, mention compactness only in
one sentence if the result is favorable.

---

## Dimension 6 — Plausibility (User Study) — Main Paper + Appendix

Per **Jacovi & Goldberg (ACL 2020)**: plausibility and faithfulness must be evaluated separately, because plausible-but-unfaithful explanations are the typical XAI failure mode.

User study reports on plausibility only (not faithfulness):
- Likert ratings on **Understandability** (5-point) and **Non-stationarity Awareness** (5-point) per explanation type
- Optional: **Calibrated Trust** — users predict whether the planner's decision is correct after reading the explanation; report prediction accuracy. This replaces raw "Trust" Likert (which incentivizes high trust regardless of correctness).
- Sample size, demographics, IRB statement reported.

**Drop**: raw Trust Likert.

**Main paper**:
- Include the user study because it directly supports whether users understand
  non-stationary explanations.
- Keep the table small: understandability, non-stationarity awareness, and
  calibrated trust / decision prediction if available.
- Clearly state that this evaluates plausibility and usefulness, not
  faithfulness.

**Appendix**:
- Full study protocol.
- Recruitment, demographics, compensation, IRB / exemption statement.
- Exact tasks, stimuli, screenshots, randomization, exclusion criteria.
- Statistical tests, confidence intervals, effect sizes, and reliability
  analysis such as ICC for Likert ratings.
- Free-response coding if collected.

---

## Ablations — Main Paper + Appendix

The ablations should not all be downgraded to the appendix. For NeurIPS, the
main paper should include a compact ablation table showing which framework
components are necessary.

### Ablation input groups

The full non-stationary explanation input has three blocks:
- **Algorithm State**
- **PCTL Analysis Results**
- **Derived Metrics Summary**

For ablation accounting, **Algorithm State is treated as part of the derived
metrics channel**, because it summarizes derived operational/non-stationary
state such as environment-change status, algorithm phase, and planner
confidence rather than raw PCTL formula values.

Therefore:
- **A1 / B4 — Single-snapshot**: keep current-snapshot PCTL values and
  current-snapshot derived metrics. Remove previous/reference-snapshot values,
  temporal deltas, recovery/turning intervals, and temporal algorithm-state
  leak fields. Current planner confidence may remain as a current-derived
  signal.
- **A2 — PCTL only, no derived metrics**: keep PCTL Analysis Results only.
  Remove both Algorithm State and Derived Metrics Summary.
- **A3 — Derived metrics only, no PCTL**: keep both Algorithm State and Derived
  Metrics Summary. Remove PCTL Analysis Results.

All ablation prompt variants should be embedded in the corresponding evaluation
generator script and derived from the same base explanation prompt
(`ns_prompt.md` / prompt id
`pmpt_6924044ca09081959632fe166b26e2bd05f29b8e9936c7f9`). The corresponding
LLM inputs must be built from the ablation's allowed evidence channels only,
rather than passing the full input and asking the model to ignore unavailable
evidence. For example, single-snapshot and PCTL-only variants should not include
previous-snapshot evidence and should not contain prompt instructions that force
a change-over-time paragraph.

**Main paper**:
- A1 — No cross-snapshot (= baseline B4)
- A2 — PCTL only, no derived metrics and no Algorithm State
- A3 — Derived metrics only, with Algorithm State but no PCTL
- A4 — **Adaptive vs fixed reference snapshot**. Tests whether comparing against the most recent previous snapshot beats comparing against a fixed initial snapshot. Non-trivial because it tests *which reference choice is correct*, not *whether more info helps*. Requires ≥3 snapshots over time (you confirmed this is your setup).

**Appendix**:
- Expanded ablation results by query type, scenario, and metric.
- Qualitative examples showing what each ablation loses.

Optional extra:
- **LLM choice robustness**: report the above metrics under at least 2 different LLMs (e.g., GPT-4, Claude). If results are stable across LLMs, framework is not LLM-specific.

**Placement for optional LLM robustness**: appendix by default; main paper only
if it becomes a reviewer-facing concern or yields an especially clean result.

---

## Timing / Efficiency — Main Short / Appendix Full

Wall-clock latency per pipeline stage (S1 query classification → S2 spec → S3 PCTL eval on S_t → S4 PCTL eval on S_t' + Δ → S5 explanation generation). Mean ± std across queries, broken down by scope (micro vs macro). Hardware setup in appendix.

**Main paper**: one sentence or a small row in the setup/results section if
space allows.

**Appendix**: full timing table, hardware, software versions, total compute,
and cost of any LLM calls.

---

## What Drops from the Current Plan

- ">95% grounding rate" target — design expectation, belongs in framework description, not evaluation.
- Cohen's kappa → use **ICC(2,1)** for Likert scores (Cohen's kappa is for categorical data).
- BERTScore as primary robustness metric — demoted to secondary.
- Raw Trust Likert — drop or replace with calibrated trust.
- "Calculate precision + recall" — undefined as written, dropped.

---

## Citation Backbone (verified)

Use these as the foundational citations for your evaluation section:

| Concept | Citation |
|---|---|
| 12-property taxonomy | Nauta et al., ACM CS 2023 ("From Anecdotal Evidence...") |
| Functionally-grounded eval | Doshi-Velez & Kim, arXiv 2017 |
| Sufficiency / Comprehensiveness | DeYoung et al., ACL 2020 (ERASER) |
| Sanity check via randomization | Adebayo et al., NeurIPS 2018 |
| Faithfulness vs Plausibility | Jacovi & Goldberg, ACL 2020 |
| Contrastive question framing | Krarup et al., JAIR 2021 |
| XRL eval landscape (general) | Milani et al., ACM CS 2024 |
| Practical metric implementations | Hedström et al., JMLR 2023 (Quantus) |

**Do not** cite specific numerical claims (X papers, Y%, Z metrics) attributed to Milani in any secondary summary — verify against the ACM CS 2024 final version directly if you need any specific Milani statistic.
