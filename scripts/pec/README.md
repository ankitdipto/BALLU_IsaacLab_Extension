# Progressive Expert Coverage (PEC)

PEC is a modular algorithm for training a mixture of specialist RL policies
that collectively cover a 2D morphology design space defined by
**Gravity Compensation Ratio (GCR)** and **Spring Coefficient (spcf)**.
Each expert is a standalone PPO policy responsible for a Gaussian-shaped region
of the design space.  Routing at deployment time is purely geometric (nearest
Gaussian centre).

---

## Algorithm Overview

```
INIT      pec_init.py
            └─ seed K Gaussians, sample N_init designs per expert
               save pec_state.json

LOOP (repeat until coverage converges)
│
├─ STEP 1  pec_train_expert.py   ×K
│            └─ draw GCR/spcf samples from expert's Gaussian
│               warm-start from previous checkpoint
│               run PPO via train.py subprocess
│               store checkpoint path in pec_state.json
│
├─ STEP 2  pec_evaluate_frontier.py
│            └─ sample F frontier / border candidates
│               for each expert: launch pec_eval_expert_frontier.py
│               record per-design curriculum level per expert
│               save candidates.json + scores.json
│
└─ STEP 3  pec_refit_gaussians.py
             └─ snapshot current Gaussians to history
                argmax-assign each frontier design to its winning expert
                MLE refit: new mu + diagonal covariance (with variance floor)
                increment iteration counter
                save updated pec_state.json
```

---

## File Reference

| Script | Role | When to run |
|--------|------|-------------|
| `pec_init.py` | Bootstrap: create state file, seed K Gaussians | Once per run |
| `pec_train_expert.py` | Train a single expert (Step 1) | K times per iteration |
| `pec_evaluate_frontier.py` | Sample frontiers & launch K evaluations (Step 2) | Once per iteration |
| `pec_eval_expert_frontier.py` | Isaac Sim subprocess: batched eval for one expert | Called by Step 2 |
| `pec_refit_gaussians.py` | Assign designs, refit Gaussians (Step 3) | Once per iteration |
| `pec_visualize.py` | Plot Gaussian mixture + frontier overlays | Anytime |
| `plot_comparison.py` | PEC oracle vs baseline comparison plots | After evaluation |
| `compare_results.py` | Quick CLI comparison of two result JSON files | Ad-hoc analysis |

---

## Shared State File — `pec_state.json`

Every script reads and/or writes `logs/pec/<run_name>/pec_state.json`.

```json
{
  "run_name":     "my_run",
  "iteration":    3,
  "usd_rel_path": "morphologies/.../robot.usd",
  "design_space": { "GCR": [0.75, 0.89], "spcf": [0.001, 0.010] },
  "experts": [
    {
      "id": 0,
      "mu": [0.79, 0.0035],
      "sigma": [[var_gcr, 0.0], [0.0, var_spcf]],
      "designs": [[gcr_0, spcf_0], ...],
      "trained": true,
      "checkpoint": "/abs/path/to/model_best.pt"
    }
  ],
  "history": [
    {
      "iteration": 0,
      "experts_snapshot": [{ "id": 0, "mu": [...], "sigma": [...],
                             "checkpoint": "...", "n_designs": 16 }]
    }
  ]
}
```

---

## Step-by-Step Usage

All commands are run from `ballu_isclb_extension/` with the `BALLU_env0`
conda environment.

### Step 0 — Initialise

```bash
conda run -n BALLU_env0 python scripts/pec/pec_init.py \
    --run_name   my_run \
    --K          3 \
    --GCR_range  0.75 0.89 \
    --spcf_range 0.001 0.010 \
    --sigma_scale 0.10 \
    --N_init     16 \
    --usd_rel_path morphologies/.../robot.usd \
    --centers 0.79 0.0035  0.85 0.0075  0.82 0.0055
```

`--centers` takes `K` GCR/spcf pairs and overrides the automatic grid
placement. Omit it to use the default grid.

**sigma_scale guidance:**

| sigma_scale | σ as % of range | Overlap at midpoint | Recommended for |
|------------|-----------------|----------------------|-----------------|
| 0.10 | ~7% | ~zero | Very peaky specialists |
| 0.15 | ~11% | very small | Moderate separation |
| 0.20 | ~14% | moderate | Broader coverage |
| 0.30 | ~21% | large | Wide initial regions |

### Step 1 — Train each expert

Run once per expert per PEC iteration:

```bash
conda run -n BALLU_env0 python scripts/pec/pec_train_expert.py \
    --run_name       my_run \
    --expert_id      0 \
    --max_iterations 1000 \
    --num_envs       4096 \
    --dl             0 \
    --headless
```

Key arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--expert_id` | required | Expert index (0-based) |
| `--max_iterations` | 2000 | PPO gradient steps |
| `--num_envs` | 4096 | Parallel sim environments |
| `--dl` | 0 | Starting obstacle difficulty level |
| `--n_samples` | = num_envs | GCR/spcf samples drawn from the Gaussian |

Warm-starting is automatic: if the expert already has a checkpoint in
`pec_state.json` it is passed to `train.py` via `--resume_path`.

### Step 2 — Evaluate frontier designs

```bash
conda run -n BALLU_env0 python scripts/pec/pec_evaluate_frontier.py \
    --run_name        my_run \
    --F               100 \
    --sampling_mode   border \
    --border_inner_ld -0.5 \
    --border_outer_ld -3.0 \
    --num_episodes    15 \
    --start_difficulty 21 \
    --headless
```

**Sampling modes:**

| Mode | Selects | Best for |
|------|---------|----------|
| `border` | Designs in an annular band 1–2.5σ from nearest Gaussian | Early iterations (expand outward gradually) |
| `frontier` | Designs with lowest max log-density (farthest from all experts) | Later iterations (fill uncovered regions) |
| `auto` | `border` for iter < `--auto_switch_iter`, then `frontier` | Fully automatic schedule |

Outputs saved to `logs/pec/<run_name>/frontier_evals/iter_<N>/`:
- `candidates.json` — the F sampled designs
- `expert_<k>_results.json` — per-design scores from each expert
- `scores.json` — aggregated scores matrix

### Step 3 — Refit Gaussians

```bash
conda run -n BALLU_env0 python scripts/pec/pec_refit_gaussians.py \
    --run_name      my_run \
    --min_var_scale 0.01
```

This script:
1. Snapshots current Gaussian parameters to `pec_state.json["history"]`
2. Assigns each frontier design to the expert with the highest score
   (ties broken by lower expert ID)
3. Recomputes each expert's `mu` (mean of all assigned designs) and
   `sigma` (diagonal covariance, with a variance floor)
4. Increments `iteration` and saves `pec_state.json`

---

## Visualisation

```bash
# Current Gaussian state
conda run -n BALLU_env0 python scripts/pec/pec_visualize.py \
    --run_name my_run \
    --output   logs/pec/my_run/plots/current.png

# Historical state at iteration 2 with frontier overlay
conda run -n BALLU_env0 python scripts/pec/pec_visualize.py \
    --run_name my_run --itr 2 \
    --output   logs/pec/my_run/plots/iter_2_frontier.png

# Historical state at iteration 2, NO frontier (clean Gaussians only)
conda run -n BALLU_env0 python scripts/pec/pec_visualize.py \
    --run_name my_run --itr 2 --no_frontier \
    --output   logs/pec/my_run/plots/iter_2_gaussians.png

# Suppress the dashed 2σ ellipse
conda run -n BALLU_env0 python scripts/pec/pec_visualize.py \
    --run_name my_run --no_2sigma \
    --output   logs/pec/my_run/plots/clean.png
```

The plot shows:
- Background heatmap — dominant expert colour fading to white in uncovered zones
- Solid 1σ ellipse (and optional dashed 2σ ellipse) per expert
- Expert centre markers (`+`)
- Training design scatter (circles)
- Frontier candidates (stars) coloured by winning expert, labelled with
  curriculum level score (only when `--itr` is given and `--no_frontier`
  is not set)

---

## Comparison Plots (PEC vs Baseline)

After evaluating both PEC experts and a baseline controller on a shared
design set (see `pec_eval_expert_frontier.py`):

```bash
conda run -n BALLU_env0 python scripts/pec/plot_comparison.py \
    --baseline   logs/pec/my_run/baseline_results.json \
    --experts    logs/pec/my_run/expert0_results.json \
                 logs/pec/my_run/expert1_results.json \
                 logs/pec/my_run/expert2_results.json \
    --pec_state  logs/pec/my_run/pec_state.json \
    --run_name   my_run \
    --output_dir logs/pec/my_run/plots
```

Generates four plots:
1. `comparison_distributions.png` — score histograms (Baseline / PEC / per-expert)
2. `comparison_diff_histogram.png` — improvement distribution (PEC − Baseline)
3. `comparison_win_breakdown.png` — win/tie/loss bar + expert win-share pie chart
4. `comparison_spatial.png` — spatial scatter: winning expert map + improvement heatmap

The evaluation input JSON must be a list of:
```json
[{"id": 0, "GCR": 0.82, "spcf": 0.005}, ...]
```
Generate a uniform sample with:
```bash
conda run -n BALLU_env0 python3 -c "
import json, numpy as np
rng = np.random.default_rng(42)
candidates = [{'id': i, 'GCR': float(g), 'spcf': float(s)}
              for i, (g, s) in enumerate(zip(
                  rng.uniform(0.75, 0.89, 1000),
                  rng.uniform(0.001, 0.010, 1000)))]
json.dump(candidates, open('eval_1000.json','w'), indent=2)
"
```

---

## Log Directory Layout

```
logs/pec/<run_name>/
├── pec_state.json                   # master state (updated each step)
├── all_expert_designs.json          # all designs across experts (for eval)
├── eval_1000_uniform.json           # optional uniform eval grid
├── baseline_eval_results.json       # baseline evaluation results
├── expert<k>_iter<n>_results.json   # per-expert evaluation results
├── frontier_evals/
│   ├── iter_0/
│   │   ├── candidates.json
│   │   ├── scores.json
│   │   ├── assignments.json         # written by pec_refit_gaussians.py
│   │   ├── expert_0_results.json
│   │   └── expert_1_results.json
│   └── iter_1/
│       └── ...
└── plots/
    ├── current_state.png
    ├── iter_<N>_frontier.png
    ├── comparison_distributions.png
    ├── comparison_diff_histogram.png
    ├── comparison_win_breakdown.png
    └── comparison_spatial.png
```

---

## Dependencies

All scripts require the `BALLU_env0` conda environment with:
- Isaac Lab (forked) + Isaac Sim 4.5.0
- RSL-RL (forked) with PPO
- `numpy`, `matplotlib`

The robot kinematic USD is resolved via the `BALLU_USD_REL_PATH` environment
variable, which is automatically injected by `pec_train_expert.py` and
`pec_evaluate_frontier.py` from `pec_state.json["usd_rel_path"]`.
