# NS-XAI: Cross-Snapshot Explanations for Adaptive Planning

Reference implementation for the NeurIPS 2026 submission *"Cross-Snapshot
Explanations for Adaptive Planning in Non-Stationary Environments."* The
framework instantiates cross-snapshot explanation on ADA-MCTS for paratransit
vehicle dispatching.

## Repository layout

```
ns_explainer/
├── core/                       NS-XAI framework
│   ├── query_classification.py     query → query type
│   ├── formula_evaluator.py        formal specification construction
│   ├── pctl_checking.py            PCTL model checking
│   ├── explanation_generation.py   evidence-grounded LLM generation
│   └── orchestrator.py             end-to-end pipeline
│
├── algo/
│   └── ADA-MCTS-main/          Adaptive MCTS planner
│
├── use_cases/
│   └── paratransit/            Paratransit instantiation
│       ├── ada_mcts_adapter.py     planner ↔ framework adapter
│       ├── pa_pctl_mapping.py      atomic propositions & derived metrics
│       ├── adamcts_runner.py       snapshot generation
│       └── config.py
│
├── logiex_baseline/            Single-snapshot LogiEx baseline
│
├── evaluation/                 Reproduction scripts for all reported tables
│   ├── query_reliability/          Table 1 (classification accuracy)
│   ├── faithfulness/               Numerical faithfulness (Appendix)
│   ├── delta_recovery/             Table 2 (Δ-Recovery)
│   ├── robustness/                 Paraphrase robustness (Appendix)
│   ├── efficiency/                 Compactness & latency (Appendix)
│   ├── baselines/                  B1, B2
│   ├── ablations/                  A2, A3, A4
│   └── tables/                     Table-generation utilities
│
├── main_paratransit_demo.py    End-to-end demonstration script
├── requirements.txt            NS-XAI environment dependencies
└── README.md
```

## Setup

The framework and the planner have different Python and dependency
requirements, so two conda environments are needed:

- `adamcts38` (Python 3.8) — runs the ADA-MCTS planner and produces planning
  snapshots.
- `xai39` (Python 3.9) — runs the NS-XAI framework, evaluation scripts, and
  `main_paratransit_demo.py`.

```bash
# ADA-MCTS environment
conda create -n adamcts38 python=3.8 -y
conda activate adamcts38
pip install -r algo/ADA-MCTS-main/requirements.txt

# NS-XAI environment
conda create -n xai39 python=3.9 -y
conda activate xai39
pip install -r requirements.txt
```

`main_paratransit_demo.py` checks for both environments at startup and exits
with instructions if either is missing.

Set your OpenAI API key (used by query classification, explanation generation,
and the evaluation graders):

```bash
export OPENAI_API_KEY=sk-...
# or place it in a .env file at the repo root: OPENAI_API_KEY=sk-...
```

## Data

The paratransit dataset (`use_cases/paratransit/data/`) is **not included** in
this release. It is publicly available from
<https://github.com/smarttransit-ai/iccps-2022-paratransit-public>; place the
contents in `use_cases/paratransit/data/`.

## Reproducing the results

The evaluation scripts produce the numbers reported in the paper. Each
sub-directory of `evaluation/` corresponds to one result set.

| Reported result | Script |
|---|---|
| Query classification accuracy (Table 1) | `evaluation/query_reliability/` |
| Numerical faithfulness (Appendix Table) | `evaluation/faithfulness/` |
| Δ-Recovery (Table 2) | `evaluation/delta_recovery/` |
| Paraphrase robustness (Appendix Table) | `evaluation/robustness/` |
| Compactness and latency (Appendix Table) | `evaluation/efficiency/` |
| Baselines (B1, B2) | `evaluation/baselines/` |
| Ablations (A2, A3, A4) | `evaluation/ablations/` |

End-to-end demo on a single scenario:

```bash
python main_paratransit_demo.py
```

### Note on runtime and caching

Running ADA-MCTS to produce planning snapshots is computationally expensive
(several hours per scenario). Intermediate snapshot caches (`pkl_cache/`,
`mdp_tn_cache/`) are excluded from the repository and are regenerated on the
first run of the corresponding script.

## License

Released under the MIT License for non-commercial research use. Code from
`algo/ADA-MCTS-main/` is included in accordance with its upstream license.
