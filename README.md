# CAIL: Change-Aware In-Context Learner

Daniel Caballero  
Independent Researcher  
danielcaballero123321@gmail.com

Code for the paper *Mitigating Shortcut Learning through Change-Aware Losses for ARC-AGI*.

CAIL is a 23M-parameter network that infers an ARC-AGI-2 task from its demonstrations in a single forward pass, without test-time training or a language model. It is trained with a change-aware loss that corrects the shortcuts a uniform cross-entropy encourages.

## Setup

```bash
pip install -r requirements.txt
```

Training expects a GPU. The results in the paper were obtained on an RTX PRO 6000 in Google Colab, using bfloat16.

## Data

Place the datasets under one directory:

```
data/
  arc-agi-2/data/training/*.json
  arc-agi-2/data/evaluation/*.json
  nvarc_training/<family>/*.json
  nvarc_full/<family>/*.json
```

ARC-AGI-2 comes from [arcprize/ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) and the synthetic tasks from the [NVARC Synthetic Puzzles](https://www.kaggle.com/datasets/sorokin/nvarc-synthetic-puzzles) dataset of [NVARC](https://github.com/1ytic/NVARC).

## Training

```bash
python cail.py --config configs/final_model.json --data-dir data --out-dir runs
```

Each run writes to `<out-dir>/<run-name>/`:

| File | Content |
|---|---|
| `meta.json` | Configuration, parameter and task counts, GPU |
| `train.jsonl` | One row per epoch |
| `eval.jsonl` | One row per evaluation, with the indices of the solved test inputs |
| `best.pt`, `last.pt` | Checkpoints: best CC and last epoch |

An interrupted run resumes from `last.pt` when launched again with the same `--out-dir`. Metric names follow the paper: `PA`, `CA`, `CC`, `UA`, `CC_background`, `CC_nonbackground` and `changed_cells_same_shape`. Metrics without a suffix are pooled over the whole evaluation set (1,000 test inputs, 2,000 for `kaggle_submission`); per-subset values end in `_arc`, `_nvarc`, `_families` and `_full`.

## Voting

```bash
python vote.py --run-dir runs/final_model --data-dir data --n-colors 125
```

This evaluates a trained run under 1,000 views (8 symmetries and 125 color permutations, with a fixed seed), and writes `vote_summary.json` and `vote_items.jsonl` to the run directory.

## Pretrained weights

The weights of the two models in the paper are in the [v1.0 release](https://github.com/danielcaballero123321-hub/cail/releases/tag/v1.0), as checkpoints that `vote.py` reads directly:

| File | Model |
|---|---|
| `final_model.pt` | The reported model, at its last epoch |
| `kaggle_submission.pt` | The model submitted to Kaggle, at its last epoch |

```bash
mkdir -p runs/final_model
curl -L -o runs/final_model/last.pt https://github.com/danielcaballero123321-hub/cail/releases/download/v1.0/final_model.pt
python vote.py --run-dir runs/final_model --data-dir data --n-colors 125
```


## Reproducing the public-evaluation score

Training on a GPU is not bitwise reproducible, so a new run of `final_model` may not reproduce the 1.25% (1.5 of 120 tasks) at its last epoch. In the run reported in the paper, the vote over 1,000 views solved two test inputs:

| Task | Test input | Votes for the correct grid | Single forward pass |
|---|---|---|---|
| `71e489b6` | 2 of 2 | 317 of 1,000 | 1 wrong cell |
| `aa4ec2a5` | 1 of 1 | 35 of 1,000 (802 distinct grids) | 73 wrong cells |

`71e489b6` is the more stable of the two: its second test input was also solved in a single forward pass at 13 evaluation epochs between 356 and 398. `aa4ec2a5` depends on the vote and is more likely to change between runs. We expect a new run to come close on these tasks, and possibly to solve them at some epoch or in some view, but this is not guaranteed.

## Configurations

Every result in the paper comes from one of the files in `configs/`, except the counting features of the Failed Ideas section, whose code is not included.

| Paper | Configurations |
|---|---|
| Table 2 (loss variants) | `uniform_ce`, `zone_weights`, `color_balance_only`, `change_aware_alpha_1.0`, `change_aware_alpha_0.93`, `change_aware_alpha_0.75`, `change_aware_alpha_0.5` |
| Optimizer | `adamw_uniform`, `adamw_alpha_0.5`, `adamw_uniform_wd_2.14`, `adamw_lr_*_epoch12_of25` |
| Scaling | `scaling_25_epochs`, `scaling_50_epochs`, `change_aware_alpha_0.93`, `scaling_200_epochs`, `final_model`, `alpha_0.5_25_epochs`, `uniform_ce_25_epochs` |
| Final model and Kaggle | `final_model`, `kaggle_submission` |
| Appendix A, run-to-run variation | `scaling_25_epochs`, `noise_seed_1`, `noise_rerun` |
| Appendix A, design decisions | `design_*` |
| Failed ideas | `failed_*` |

`adamw_lr_*_epoch12_of25` train only the first epochs of a 25-epoch schedule (`epochs` is lower than `sched_epochs`).

The configurations named `design_diagonal_rope`, `design_no_1d_weight_decay` and `design_1d_weight_decay` belong to earlier versions of the model, evaluated without NVARC full in the paper. `failed_bottleneck_*` and `failed_no_bottleneck` use the same earlier architecture.

Training on a GPU is not bitwise reproducible, even with the same seed, and some configurations vary widely between runs.

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

---

The author was 17 years old when this work was done.

A personal note from the author: [A_NOTE.md](A_NOTE.md).
