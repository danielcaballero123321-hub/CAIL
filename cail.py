"""CAIL: the Change-Aware In-Context Learner for ARC-AGI-2.

Training, evaluation and model code for the paper
"Mitigating Shortcut Learning through Change-Aware Losses for ARC-AGI".

Usage: python cail.py --config configs/final_model.json --data-dir data --out-dir runs

Each run writes meta.json, train.jsonl, eval.jsonl, best.pt and last.pt to <out-dir>/<run-name>/.
Metric names follow the paper: PA, CA, CC, UA, CC_background, CC_nonbackground, changed_cells.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
import zlib
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


EMPTY_VALUE = 10
N_COLORS = 11
GRID_SIZE = 32


PIXEL_CLASSES = ("changed_color", "changed_empty", "unchanged_color", "unchanged_empty")


@dataclass
class Config:
    """Run, data, architecture, optimizer and loss configuration."""

    run_name: str = "cail"
    out_dir: str = "runs"

    save_ckpt: bool = True

    sched_epochs: int = 0

    arc_train_dir: str = "data/arc-agi-2/data/training"
    arc_eval_dir: str = "data/arc-agi-2/data/evaluation"
    nvarc_dir: str = "data/nvarc_training"
    use_nvarc: bool = False

    nvarc_heldout_n: int = 250
    nvarc_split_seed: int = 1235

    nvarc_families_heldout: int = 10
    nvarc_families_seed: int = 5679

    nvarc_tasks_per_family: int = 50

    nvarc_full_dir: str = "data/nvarc_full"
    nvarc_full_eval_n: int = 350
    nvarc_full_seed: int = 2468

    train_with_nvarc_full: bool = False

    min_pairs: int = 3
    max_pairs: int = 6

    min_pairs_train: int = 3

    use_color_perm: bool = True

    use_flips: bool = True
    use_rotations: bool = True

    d_model: int = 256
    n_heads: int = 4
    head_dim: int = 64
    ffn_mult: int = 2
    dropout: float = 0.0

    attn_gate: str = ""
    attn_gate_adamw: bool = False
    rope_base: float = 10.0

    rope_a_js: tuple = ()

    rope_a3_k: bool = False

    rope_b4_q: bool = False
    rope_b4_js: tuple = ()

    rope_diag: bool = False

    rope_k_a3_q: bool = False
    rope_k_b4_k: bool = False

    rope_b3: bool = False

    rope_k_self: bool = False

    layerscale_init: float = 0.1
    n_bottleneck_tokens: int = 4

    cnn_in_stages: tuple = (128, 256, 256)
    cnn_in_depths: tuple = (1, 1, 1)
    cnn_in_downsamples: tuple = (1, 2)
    cnn_out_stages: tuple = (128, 256, 256)
    cnn_out_depths: tuple = (1, 1, 1)
    cnn_out_downsamples: tuple = (1, 2)

    cnn_demo_sees_pair: bool = False

    cnn_demo_shared: bool = False

    cnn_demo_sees_test: bool = False

    cnn_demo_stem: bool = False
    cnn_demo_stem_depth: int = 2

    cnn_demo_attn: bool = False

    cnn_demo_pw_dw: int = 0

    grid_cross_before: int = -1
    cnn_grid_stages: tuple = (128, 256, 256)
    cnn_grid_depths: tuple = (1, 1, 1)
    cnn_grid_downsamples: tuple = (1, 2)
    cnn_grid_block_type: str = ""

    cnn_test_stages: tuple = (256, 256)
    cnn_test_depths: tuple = (2, 2)
    cnn_test_downsamples: tuple = ()

    cnn_test_v41_expansion: int = 1

    block_type: str = "v41"
    block_norm: str = "group"
    downsample_norm: str = "groupnorm"

    layer_scale_init_cnn: float = 0.1

    cnn_in_block_type: str = ""
    cnn_out_block_type: str = ""
    cnn_test_block_type: str = ""

    convnext_expansion: int = 4

    n_blocks_a2: int = 4
    n_blocks_a3: int = 2
    n_blocks_b2: int = 2
    n_blocks_b3: int = 1
    n_blocks_b4: int = 4

    b4_self: bool = True
    b4_cross_k: bool = False

    k_order: tuple = ()

    bridge: str = "k"

    decoder: str = "b4"
    pixel_self_every: int = 0

    demos_read_test: bool = True

    b4_pixelattn: bool = False

    b4_pixelattn_every: int = 1

    rope_px_self: str = "b"

    ffn_px_dw: int = 0

    b4_rounds: int = 1

    cond_test: str = ""

    attn_res: bool = False

    attn_res_projected_query: bool = False

    attn_res_query_from: str = "sum"

    batch_size: int = 32
    lr: float = 8e-5

    lr_min: float = 3e-8

    lr_shape: str = "cosine"
    warmup_epochs: float = 20
    weight_decay: float = 1.0

    optimizer_name: str = "adamw"

    decay_1d: bool = True
    epochs: int = 1200

    grad_clip: float = 10.0
    sched_every: int = 31

    num_workers: int = 2
    pin_memory: bool = True
    use_amp: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    loss_class_weights: tuple = (1.0, 1.0, 1.0, 1.0)

    loss_beta_start: float = 0.0
    loss_beta_end: float = 1.0

    color_balanced_loss: bool = False
    loss_zone_weights: tuple = (1.0, 1.0)

    loss_alpha: float = 1.0
    loss_zone_weights_gen: tuple = (1.0, 1.0)

    empty_weight: float = 1.0

    loss_task_mass: bool = False

    free_frame: bool = False

    eval_every: int = 50
    eval_seed: int = 4321

    seed: int = 0
    verify_pipeline: bool = True

    compile_model: bool = True

    progress_line: bool = True

    @property
    def run_dir(self) -> str:
        return os.path.join(self.out_dir, self.run_name)

    @property
    def save_path(self) -> str:
        return os.path.join(self.run_dir, "best.pt")

    @property
    def backup_path(self) -> str:
        return os.path.join(self.run_dir, "last.pt")

    @property
    def max_demos(self) -> int:
        return self.max_pairs - 1

    @property
    def lr_floor(self) -> float:
        return self.lr if self.lr_min == 0 else self.lr_min


def _hw_final(downsamples: tuple) -> int:
    return GRID_SIZE // (2 ** len(downsamples))


def validate(cfg: Config):
    """Rejects inconsistent configurations before training starts."""
    if cfg.d_model % 2:
        raise ValueError("Invalid configuration: d_model")
    if cfg.bridge not in ("k", "demos"):
        raise ValueError("Invalid configuration: puente")
    no_bottleneck = cfg.bridge == "demos"
    if no_bottleneck:
        extra_ = [
            n
            for n, v in (
                ("n_bottleneck_tokens", cfg.n_bottleneck_tokens),
                ("n_blocks_a3", cfg.n_blocks_a3),
                ("rope_a3_k", cfg.rope_a3_k),
                ("rope_k_a3_q", cfg.rope_k_a3_q),
                ("rope_k_b4_k", cfg.rope_k_b4_k),
                ("rope_k_self", cfg.rope_k_self),
                ("k_order", tuple(cfg.k_order)),
                ("b4_cross_k", cfg.b4_cross_k),
            )
            if v
        ]
        if extra_:
            raise ValueError("Invalid argument or configuration.")
    if cfg.decoder not in ("b4", "two_way"):
        raise ValueError("Invalid configuration: decoder")
    if cfg.decoder == "two_way":
        if not no_bottleneck:
            raise ValueError("Invalid argument or configuration.")
        if cfg.b4_self:
            raise ValueError("Invalid configuration: b4_self")
    elif not cfg.demos_read_test:
        raise ValueError("Invalid configuration: demos_read_test")
    if cfg.train_with_nvarc_full and not cfg.use_nvarc:
        raise ValueError("Invalid configuration: train_with_nvarc_full, use_nvarc")
    if cfg.optimizer_name not in ("adamw", "muon"):
        raise ValueError("Invalid configuration: optimizer_name")
    if cfg.cond_test not in ("", "mean", "cross"):
        raise ValueError("Invalid configuration: cond_test")
    if cfg.cnn_test_v41_expansion < 1 or (
        cfg.cnn_test_v41_expansion > 1 and block_type_of(cfg, "test") != "v41"
    ):
        raise ValueError("Invalid configuration: cnn_test_v41_expansion")
    if cfg.b4_rounds < 1 or (cfg.b4_rounds > 1 and cfg.bridge != "demos"):
        raise ValueError("Invalid configuration: b4_rounds, puente")
    if cfg.attn_res and (cfg.bridge != "demos" or cfg.b4_rounds != 1):
        raise ValueError("Invalid configuration: attn_res, b4_rounds, puente")
    if cfg.attn_res_projected_query and not cfg.attn_res:
        raise ValueError("Invalid configuration: attn_res, attn_res_projected_query")
    if cfg.attn_res_query_from not in ("sum", "last"):
        raise ValueError("Invalid configuration: attn_res_query_from")
    if cfg.attn_res_query_from != "sum" and not cfg.attn_res_projected_query:
        raise ValueError("Invalid configuration: attn_res_query_from, attn_res_projected_query")
    if cfg.b4_pixelattn and cfg.decoder != "two_way":
        raise ValueError("Invalid configuration: b4_pixelattn, decoder")
    if cfg.rope_px_self not in ("b", "b4"):
        raise ValueError("Invalid configuration: rope_px_self")
    if cfg.rope_px_self == "b4" and not (
        cfg.b4_pixelattn or cfg.pixel_self_every or (cfg.b4_self and cfg.decoder != "two_way")
    ):
        raise ValueError(
            "Invalid configuration: b4_pixelattn, b4_self, decoder, pixel_self_every, rope_px_self"
        )
    if cfg.b4_pixelattn_every < 1 or (cfg.b4_pixelattn_every > 1 and not cfg.b4_pixelattn):
        raise ValueError("Invalid configuration: b4_pixelattn, b4_pixelattn_every")
    if cfg.cnn_demo_pw_dw < 0 or (cfg.cnn_demo_pw_dw and cfg.cnn_demo_pw_dw % 2 == 0):
        raise ValueError("Invalid configuration: cnn_demo_pw_dw")
    if cfg.cnn_demo_pw_dw:
        demo = ("in",) if cfg.cnn_demo_shared else ("in", "out")
        others = [n for n in demo if block_type_of(cfg, n) != "convnext_v1"]
        if others:
            raise ValueError("Invalid argument or configuration.")
    if cfg.ffn_px_dw < 0 or (cfg.ffn_px_dw and cfg.ffn_px_dw % 2 == 0):
        raise ValueError("Invalid configuration: ffn_px_dw")
    if cfg.ffn_px_dw and cfg.decoder != "two_way":
        raise ValueError("Invalid configuration: decoder, ffn_px_dw")
    if cfg.pixel_self_every < 0 or (cfg.pixel_self_every and cfg.decoder != "two_way"):
        raise ValueError("Invalid configuration: decoder, pixel_self_every")
    if cfg.k_order:
        o = tuple(cfg.k_order)
        if set(o) - {"a", "b"} or o.count("a") != cfg.n_blocks_a3 or o.count("b") != cfg.n_blocks_b3:
            raise ValueError("Invalid configuration: n_blocks_a3, n_blocks_b3")
    if cfg.head_dim % 2:
        raise ValueError("Invalid configuration: head_dim")
    if cfg.attn_gate not in ("", "elementwise", "headwise"):
        raise ValueError("Invalid configuration: attn_gate")
    if cfg.attn_gate_adamw and not cfg.attn_gate:
        raise ValueError("Invalid configuration: attn_gate, attn_gate_adamw")
    diag = ("diagonal", "antidiagonal") if cfg.rope_diag else ()
    axes_a = ("row", "column", "grid") + diag
    if cfg.head_dim // 2 < len(axes_a):
        raise ValueError("Invalid configuration: head_dim")
    if cfg.rope_a_js:
        r = (cfg.head_dim // 2) // len(axes_a)
        if len(cfg.rope_a_js) != len(axes_a):
            raise ValueError("Invalid configuration: rope_a_js")
        for axis_, js in zip(axes_a, cfg.rope_a_js):
            js = tuple(js)
            if any(not isinstance(j, int) or not 0 <= j < r for j in js):
                raise ValueError("Invalid argument or configuration.")
            if len(set(js)) != len(js):
                raise ValueError("Invalid argument or configuration.")
    axes_b = ("row", "column") + diag
    if cfg.rope_b4_js:
        r = (cfg.head_dim // 2) // len(axes_b)
        if len(cfg.rope_b4_js) != len(axes_b):
            raise ValueError("Invalid configuration: rope_b4_js")
        for axis_, js in zip(axes_b, cfg.rope_b4_js):
            js = tuple(js)
            if any(not isinstance(j, int) or not 0 <= j < r for j in js):
                raise ValueError("Invalid argument or configuration.")
            if len(set(js)) != len(js):
                raise ValueError("Invalid argument or configuration.")

    if not no_bottleneck and (cfg.rope_k_a3_q or cfg.rope_k_b4_k or cfg.rope_b3 or cfg.rope_k_self):
        hw = _hw_final(cfg.cnn_in_downsamples)
        if cfg.n_bottleneck_tokens != hw * hw:
            raise ValueError("Invalid configuration: n_bottleneck_tokens")
    if cfg.min_pairs_train and not cfg.min_pairs <= cfg.min_pairs_train <= cfg.max_pairs:
        raise ValueError("Invalid configuration: max_pairs, min_pairs, min_pairs_train")
    if not 3 <= cfg.min_pairs <= cfg.max_pairs:
        raise ValueError("Invalid configuration: max_pairs, min_pairs")

    with_grid = cfg.grid_cross_before != -1
    for name_ in ("in", "out", "test") + (("grid",) if with_grid else ()):
        st = getattr(cfg, f"cnn_{name_}_stages")
        dp = getattr(cfg, f"cnn_{name_}_depths")
        ds = getattr(cfg, f"cnn_{name_}_downsamples")
        if len(st) != len(dp):
            raise ValueError("Invalid argument or configuration.")
        if any(i >= len(st) for i in ds):
            raise ValueError("Invalid argument or configuration.")
        for i in range(1, len(st)):
            if i not in ds and st[i] != st[i - 1]:
                raise ValueError("Invalid argument or configuration.")
        if _hw_final(ds) < 1:
            raise ValueError("Invalid argument or configuration.")

    if cfg.cnn_demo_shared and not cfg.cnn_demo_sees_pair:
        raise ValueError("Invalid configuration: cnn_demo_shared, cnn_demo_sees_pair")

    if not cfg.cnn_demo_shared and _hw_final(cfg.cnn_in_downsamples) != _hw_final(cfg.cnn_out_downsamples):
        raise ValueError("Invalid configuration: cnn_demo_shared, cnn_in_downsamples, cnn_out_downsamples")

    if with_grid:
        if not 0 <= cfg.grid_cross_before < cfg.n_blocks_b4:
            raise ValueError("Invalid configuration: grid_cross_before, n_blocks_b4")
        if not no_bottleneck or cfg.b4_rounds != 1 or cfg.attn_res:
            raise ValueError("Invalid configuration: attn_res, b4_rounds")
        if _hw_final(cfg.cnn_grid_downsamples) != _hw_final(cfg.cnn_in_downsamples):
            raise ValueError("Invalid configuration: cnn_grid_downsamples, cnn_in_downsamples")

    if cfg.cnn_demo_sees_test and (not cfg.cnn_demo_sees_pair or cfg.cnn_demo_stem):
        raise ValueError("Invalid configuration: cnn_demo_stem, cnn_demo_sees_pair, cnn_demo_sees_test")

    if cfg.cnn_demo_stem:
        if not cfg.cnn_demo_sees_pair:
            raise ValueError("Invalid configuration: cnn_demo_sees_pair")
        if not cfg.cnn_demo_shared and cfg.cnn_in_stages[0] != cfg.cnn_out_stages[0]:
            raise ValueError("Invalid configuration: cnn_demo_shared, cnn_in_stages, cnn_out_stages")
        if cfg.cnn_in_stages[0] % 2:
            raise ValueError("Invalid configuration: cnn_in_stages")
        if cfg.cnn_demo_stem_depth < 0:
            raise ValueError("Invalid configuration: cnn_demo_stem_depth")

    if cfg.cnn_demo_attn:
        for name_ in ("in",) if cfg.cnn_demo_shared else ("in", "out"):
            st = getattr(cfg, f"cnn_{name_}_stages")
            ds = getattr(cfg, f"cnn_{name_}_downsamples")
            if not ds:
                raise ValueError("Invalid argument or configuration.")
            width = st[max(min(ds) - 1, 0)]
            if width % cfg.head_dim:
                raise ValueError("Invalid configuration: head_dim")
    for _t in (cfg.block_type, *(block_type_of(cfg, n) for n in ("in", "out", "test", "grid"))):
        if _t not in ("v41", "convnext_v1"):
            raise ValueError("Invalid argument or configuration.")
    if cfg.convnext_expansion < 1:
        raise ValueError("Invalid configuration: convnext_expansion")
    if not no_bottleneck and cfg.n_bottleneck_tokens < 1:
        raise ValueError("Invalid configuration: n_bottleneck_tokens")
    if cfg.sched_every < 1:
        raise ValueError("Invalid configuration: sched_every")
    if cfg.lr_min < 0:
        raise ValueError("Invalid configuration: lr_min")
    if cfg.lr_shape not in ("cosine", "linear"):
        raise ValueError("Invalid configuration: lr_shape")
    if cfg.lr_min > cfg.lr:
        raise ValueError("Invalid configuration: lr, lr_min")
    if len(cfg.loss_class_weights) != len(PIXEL_CLASSES):
        raise ValueError("Invalid configuration: loss_class_weights")
    if any(not w > 0 for w in cfg.loss_class_weights):
        raise ValueError("Invalid configuration: loss_class_weights")
    if cfg.loss_beta_start < 0 or cfg.loss_beta_end < 0:
        raise ValueError("Invalid configuration: loss_beta_end, loss_beta_start")
    for name_ in ("loss_zone_weights", "loss_zone_weights_gen"):
        pz = getattr(cfg, name_)
        if len(pz) != 2 or any(w < 0 for w in pz) or sum(pz) <= 0:
            raise ValueError("Invalid argument or configuration.")
    if not 0.0 <= cfg.loss_alpha <= 1.0:
        raise ValueError("Invalid configuration: loss_alpha")
    if cfg.loss_task_mass and not cfg.color_balanced_loss:
        raise ValueError("Invalid configuration: loss_task_mass, color_balanced_loss")
    if cfg.empty_weight < 0:
        raise ValueError("Invalid configuration: empty_weight")


def set_seed(seed: int):
    """Seeds Python, NumPy and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int):

    s = torch.initial_seed() % (2**32)
    random.seed(s)
    np.random.seed(s)


class _RngState:
    def __enter__(self):
        self._py = random.getstate()
        self._np = np.random.get_state()
        return self

    def __exit__(self, *exc):
        random.setstate(self._py)
        np.random.set_state(self._np)
        return False


def pad_grid(grid) -> np.ndarray:
    """Places a grid on the 32x32 canvas at (1, 1), filling the rest with the empty class."""
    g = np.asarray(grid, dtype=np.uint8)
    h, w = g.shape
    out = np.full((GRID_SIZE, GRID_SIZE), EMPTY_VALUE, dtype=np.uint8)
    out[1 : h + 1, 1 : w + 1] = g
    return out


def augment_pairs(pairs: list, cfg: Config) -> list:
    """Applies the same color permutation and symmetry to every pair of a task."""
    if cfg.use_color_perm:
        lut = np.arange(N_COLORS, dtype=np.uint8)
        lut[1:10] = np.random.permutation(np.arange(1, 10))
    else:
        lut = None

    do_flip_h = cfg.use_flips and random.random() > 0.5
    do_flip_v = cfg.use_flips and random.random() > 0.5
    rotation = random.choice([0, 1, 2, 3]) if cfg.use_rotations else 0

    def aug(grid):
        g = np.asarray(grid, dtype=np.uint8)
        if lut is not None:
            g = lut[g]
        if do_flip_h:
            g = np.flip(g, axis=1)
        if do_flip_v:
            g = np.flip(g, axis=0)
        if rotation:
            g = np.rot90(g, k=rotation)
        return np.ascontiguousarray(g)

    return [{"input": aug(p["input"]), "output": aug(p["output"])} for p in pairs]


def load_arc(data_dir: str, min_pairs_: int) -> list:
    """Loads ARC-AGI-2 tasks, keeping the official demonstration/test split."""
    tasks = []
    for fp in sorted(Path(data_dir).glob("**/*.json")):
        try:
            with open(fp) as f:
                task = json.load(f)
        except json.JSONDecodeError:
            continue
        ok = lambda p: "input" in p and "output" in p
        demos = [p for p in task.get("train", []) if ok(p)]
        tests = [p for p in task.get("test", []) if ok(p)]
        if demos and tests and len(demos) + len(tests) >= min_pairs_:
            tasks.append({"id": fp.stem, "pairs": demos + tests, "demos": demos, "tests": tests})
    return tasks


@dataclass
class NvarcSplit:
    train: list
    files: list
    families: list
    selected_families: list
    n_families_total: int


def families_of(nvarc_dir: str) -> dict:
    fams = {}
    for f in sorted(Path(nvarc_dir).glob("*/*.json")):
        fams.setdefault(f.parent.name, []).append(str(f))
    return fams


def _check_eval_holdout(nvarc_dir: str, arc_eval_dir: str, families: set):
    eval_ids = {f.stem for f in Path(arc_eval_dir).glob("**/*.json")}
    collision = families & eval_ids
    if collision:
        raise ValueError("Invalid argument or configuration.")


def split_nvarc(
    nvarc_dir: str,
    arc_eval_dir: str,
    heldout_n: int,
    split_seed: int,
    n_families: int = 0,
    families_seed: int = 0,
    tasks_per_family: int = 0,
) -> NvarcSplit:
    """Splits NVARC training into training tasks and the two held-out evaluation subsets."""
    fams = families_of(nvarc_dir)
    if not fams:
        raise FileNotFoundError("Dataset files not found: check the data directory.")
    _check_eval_holdout(nvarc_dir, arc_eval_dir, set(fams))

    names = sorted(fams)
    n_fam = max(0, min(int(n_families), len(names) - 1))
    selected_ = sorted(random.Random(families_seed).sample(names, n_fam))

    held_families = []
    for name_ in selected_:
        paths = fams[name_]

        if tasks_per_family and len(paths) > tasks_per_family:
            paths = [
                paths[i]
                for i in sorted(
                    random.Random(families_seed + zlib.crc32(name_.encode()) % 10_000).sample(
                        range(len(paths)), tasks_per_family
                    )
                )
            ]
        held_families.extend(paths)

    excluded = set(selected_)
    rest = [f for n, rs in sorted(fams.items()) if n not in excluded for f in rs]
    n = min(int(heldout_n), max(0, len(rest) - 1))
    idx = set(random.Random(split_seed).sample(range(len(rest)), n))
    return NvarcSplit(
        train=[f for i, f in enumerate(rest) if i not in idx],
        files=[f for i, f in enumerate(rest) if i in idx],
        families=held_families,
        selected_families=selected_,
        n_families_total=len(names),
    )


def arc_without_families(arc_tasks: list, families) -> list:
    excluded = set(families)
    return [t for t in arc_tasks if t["id"] not in excluded]


def select_nvarc_full(full_dir: str, n: int, seed: int) -> list:
    """Selects the NVARC full tasks used for evaluation."""
    if n <= 0:
        return []
    paths = [str(f) for f in sorted(Path(full_dir).glob("*/*.json"))]
    if not paths:
        raise FileNotFoundError("Dataset files not found: check the data directory.")
    idx = sorted(random.Random(seed).sample(range(len(paths)), min(n, len(paths))))
    return [paths[i] for i in idx]


def _nvarc_pairs(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def _sample_pairs(pairs: list, cfg: Config) -> tuple:
    n = random.randint(cfg.min_pairs_train or cfg.min_pairs, cfg.max_pairs)
    n = min(n, len(pairs))
    selected = random.sample(range(len(pairs)), n)
    sel = [pairs[i] for i in selected]
    return sel[:-1], sel[-1]


class TrainDataset(Dataset):
    """Samples three to six pairs per task, using the last one as the test pair."""

    def __init__(self, cfg: Config, arc_tasks: list | None = None, nvarc_files: list | None = None):
        self.cfg = cfg
        self.arc = load_arc(cfg.arc_train_dir, cfg.min_pairs) if arc_tasks is None else arc_tasks
        if nvarc_files is not None:
            self.nvarc = nvarc_files
        elif cfg.use_nvarc:
            split = split_nvarc(
                cfg.nvarc_dir,
                cfg.arc_eval_dir,
                cfg.nvarc_heldout_n,
                cfg.nvarc_split_seed,
                cfg.nvarc_families_heldout,
                cfg.nvarc_families_seed,
                cfg.nvarc_tasks_per_family,
            )
            self.nvarc = split.train

            if arc_tasks is None:
                self.arc = arc_without_families(self.arc, split.selected_families)
        else:
            self.nvarc = []

    def __len__(self):
        return len(self.arc) + len(self.nvarc)

    def __getitem__(self, idx):
        cfg = self.cfg
        if idx < len(self.arc):
            pairs = self.arc[idx]["pairs"]
        else:
            pairs = _nvarc_pairs(self.nvarc[idx - len(self.arc)])

        demos, test = _sample_pairs(pairs, cfg)
        aug = augment_pairs(demos + [test], cfg)
        demos, test = aug[:-1], aug[-1]

        return (
            torch.from_numpy(np.stack([pad_grid(p["input"]) for p in demos])),
            torch.from_numpy(np.stack([pad_grid(p["output"]) for p in demos])),
            torch.from_numpy(pad_grid(test["input"])),
            torch.from_numpy(pad_grid(test["output"])),
            0,
            idx,
        )


def collate_fn(batch):
    n_demos = [int(b[0].shape[0]) for b in batch]
    return (
        torch.cat([b[0] for b in batch], dim=0),
        torch.cat([b[1] for b in batch], dim=0),
        torch.stack([b[2] for b in batch], dim=0),
        torch.stack([b[3] for b in batch], dim=0),
        torch.tensor(n_demos, dtype=torch.long),
        torch.tensor([b[4] for b in batch], dtype=torch.long),
        torch.tensor([b[5] for b in batch], dtype=torch.long),
    )


SUBSET_ARC, SUBSET_NVARC, SUBSET_FAMILIES, SUBSET_FULL = 1, 2, 3, 4
SUBSET_SUFFIX = {
    SUBSET_ARC: "_arc",
    SUBSET_NVARC: "_nvarc",
    SUBSET_FAMILIES: "_families",
    SUBSET_FULL: "_full",
}
SUBSET_NAME = {SUBSET_ARC: "ARC", SUBSET_NVARC: "NVARC", SUBSET_FAMILIES: "FAMILIES", SUBSET_FULL: "FULL"}


SUBSETS_IN_TOTAL = (SUBSET_ARC, SUBSET_NVARC, SUBSET_FAMILIES, SUBSET_FULL)


class EvalSet(Dataset):
    """Frozen evaluation set: the same demonstrations, test pair and order at every evaluation."""

    def __init__(
        self, cfg: Config, split: "NvarcSplit | None" = None, verbose: bool = True, full: "list | None" = None
    ):
        self.cfg = cfg
        items, tasks, info = [], [], []
        trimmed = 0

        def add_item(demos, test, group, task_, record):

            record["item"] = len(items)
            items.append(
                (
                    np.stack([pad_grid(p["input"]) for p in demos]),
                    np.stack([pad_grid(p["output"]) for p in demos]),
                    pad_grid(test["input"]),
                    pad_grid(test["output"]),
                    group,
                    len(items),
                )
            )
            tasks.append(task_)
            info.append(record)

        with _RngState():
            arc = load_arc(cfg.arc_eval_dir, cfg.min_pairs)
            for i, t in enumerate(arc):
                demos = t["demos"][: cfg.max_demos]
                trimmed += len(t["demos"]) > cfg.max_demos
                for p, test in enumerate(t["tests"]):
                    add_item(
                        demos,
                        test,
                        SUBSET_ARC,
                        i,
                        {
                            "id": t["id"],
                            "family": None,
                            "subset": "ARC",
                            "pair": p,
                            "of": len(t["tests"]),
                            "n_demos": len(demos),
                            "demos_in_file": len(t["demos"]),
                        },
                    )

            if split is None and cfg.use_nvarc:
                split = split_nvarc(
                    cfg.nvarc_dir,
                    cfg.arc_eval_dir,
                    cfg.nvarc_heldout_n,
                    cfg.nvarc_split_seed,
                    cfg.nvarc_families_heldout,
                    cfg.nvarc_families_seed,
                    cfg.nvarc_tasks_per_family,
                )

            loose = list(split.files) if split else []
            family_files = list(split.families) if split else []
            self.families_heldout = list(split.selected_families) if split else []
            for j, path in enumerate(loose + family_files):
                group = SUBSET_NVARC if j < len(loose) else SUBSET_FAMILIES
                pairs = _nvarc_pairs(path)

                rng = random.Random(cfg.eval_seed + 7_000_003 + j * 10007)
                n = rng.randint(cfg.min_pairs, cfg.max_pairs)
                n = max(cfg.min_pairs, min(n, len(pairs)))

                idxs = rng.sample(range(len(pairs)), n)

                add_item(
                    [pairs[k] for k in idxs[:-1]],
                    pairs[idxs[-1]],
                    group,
                    len(arc) + j,
                    {
                        "id": Path(path).stem,
                        "family": Path(path).parent.name,
                        "subset": SUBSET_NAME[group],
                        "pair": idxs[-1],
                        "of": len(pairs),
                        "demos": idxs[:-1],
                        "n_demos": n - 1,
                    },
                )

            if full is None:
                full = select_nvarc_full(cfg.nvarc_full_dir, cfg.nvarc_full_eval_n, cfg.nvarc_full_seed)
            base = len(arc) + len(loose) + len(family_files)
            for j, path in enumerate(full):
                pairs = _nvarc_pairs(path)
                rng = random.Random(cfg.eval_seed + 9_000_011 + j * 10007)
                n = rng.randint(cfg.min_pairs, cfg.max_pairs)
                n = max(cfg.min_pairs, min(n, len(pairs)))
                idxs = rng.sample(range(len(pairs)), n)

                add_item(
                    [pairs[k] for k in idxs[:-1]],
                    pairs[idxs[-1]],
                    SUBSET_FULL,
                    base + j,
                    {
                        "id": Path(path).stem,
                        "family": Path(path).parent.name,
                        "subset": SUBSET_NAME[SUBSET_FULL],
                        "pair": idxs[-1],
                        "of": len(pairs),
                        "demos": idxs[:-1],
                        "n_demos": n - 1,
                    },
                )

        self.items = items
        self.tasks = tasks
        self.info = info
        self.n_arc = sum(1 for it in items if it[4] == SUBSET_ARC)
        self.n_nvarc = sum(1 for it in items if it[4] == SUBSET_NVARC)
        self.n_families_tasks = sum(1 for it in items if it[4] == SUBSET_FAMILIES)
        self.n_full = sum(1 for it in items if it[4] == SUBSET_FULL)
        self.n_arc_tasks = len({t for t, it in zip(tasks, items) if it[4] == SUBSET_ARC})

        if verbose:
            nb = sum(a.nbytes + b.nbytes + c.nbytes + d.nbytes for a, b, c, d, _, _ in items)
            print(
                f"Frozen eval set: {len(items)} pairs | ARC {self.n_arc} "
                f"from {self.n_arc_tasks} tasks (official protocol) | NVARC "
                f"by file {self.n_nvarc} | NVARC by family "
                f"{self.n_families_tasks} from {len(self.families_heldout)} families | "
                f"FULL {self.n_full} | "
                f"{nb / 1e6:.1f} MB | eval_seed={cfg.eval_seed}"
            )
            if self.families_heldout:
                print(f"held-out families: {', '.join(self.families_heldout)}")
            if trimmed:
                print(
                    f"                     {trimmed} ARC task(s) with more than "
                    f"{cfg.max_demos} demos: using the first "
                    f"{cfg.max_demos}"
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        a, b, c, d, g, t = self.items[i]
        return (torch.from_numpy(a), torch.from_numpy(b), torch.from_numpy(c), torch.from_numpy(d), g, t)


class _ChannelsLastConvFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, stride, padding, groups):
        ctx.save_for_backward(x, w)
        ctx.conf = (stride, padding, groups, b is not None)
        return F.conv2d(x, w, b, stride=stride, padding=padding, groups=groups)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        stride, padding, groups, with_b = ctx.conf
        g = g.contiguous(memory_format=torch.channels_last)
        asks_gx = ctx.needs_input_grad[0]
        flipped = (
            asks_gx
            and groups > 1
            and groups == x.shape[1]
            and stride == (1, 1)
            and all(2 * p + 1 == k for p, k in zip(padding, w.shape[2:]))
        )
        gx, gw, _ = torch.ops.aten.convolution_backward(
            g,
            x,
            w,
            None,
            list(stride),
            list(padding),
            [1, 1],
            False,
            [0, 0],
            groups,
            [asks_gx and not flipped, ctx.needs_input_grad[1], False],
        )
        if flipped:
            gx = F.conv2d(g, w.flip(2, 3), padding=padding, groups=groups)
        gb = g.sum((0, 2, 3), dtype=torch.float32).to(g.dtype) if with_b and ctx.needs_input_grad[2] else None
        return gx, gw, gb, None, None, None


class ChannelsLastConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.padding_mode != "zeros" or isinstance(self.padding, str) or self.dilation != (1, 1):
            raise ValueError("Invalid argument or configuration.")

    def forward(self, x):

        dt = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
        with torch.autocast(x.device.type, enabled=False):
            b = None if self.bias is None else self.bias.to(dt)
            return _ChannelsLastConvFn.apply(
                x.to(dt), self.weight.to(dt), b, self.stride, self.padding, self.groups
            )


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.num_channels = num_channels

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.num_channels,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


def make_norm(kind: str, channels: int) -> nn.Module:
    if kind == "layernorm":
        return LayerNorm2d(channels)
    if kind == "groupnorm":
        return nn.GroupNorm(1, channels)
    raise ValueError("Invalid argument or configuration.")


def make_block_norm(kind: str, channels: int, eps: float) -> tuple:
    if kind == "layer":
        return nn.LayerNorm(channels, eps=eps), False
    if kind == "group":
        return nn.GroupNorm(1, channels, eps=eps), True
    raise ValueError("Invalid argument or configuration.")


class V41Block(nn.Module):
    """Convolutional block with dense 3x3 layers instead of pointwise layers."""

    def __init__(
        self, dim: int, out_dim: int = None, block_norm: str = "group", cond_dim: int = 0, expansion: int = 1
    ):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        h = expansion * out_dim
        self.dw = ChannelsLastConv2d(dim, dim, 7, padding=3, groups=dim, bias=False)
        self.norm, self.norm_nchw = make_block_norm(block_norm, dim, 1e-5)
        self.pw1 = ChannelsLastConv2d(dim, h, 3, padding=1, bias=False)
        self.pw2 = ChannelsLastConv2d(h, out_dim, 3, padding=1, bias=False)
        self.skip = nn.Identity() if dim == out_dim else ChannelsLastConv2d(dim, out_dim, 1, bias=False)

        self.ada = nn.Linear(cond_dim, 2 * dim) if cond_dim else None

    def forward(self, x, cond=None):
        r = self.skip(x)
        x = self.dw(x)
        if self.norm_nchw:
            x = self.norm(x)
        else:
            x = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        if cond is not None:
            g, b = self.ada(cond).chunk(2, dim=-1)
            x = x * (1 + g[:, :, None, None]) + b[:, :, None, None]
        x = F.gelu(self.pw1(x))
        return self.pw2(x) + r


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block with group normalization and optional AdaGN conditioning."""

    def __init__(self, dim: int, out_dim: int = None, expansion: int = 4, pw_dw: int = 0, cond_dim: int = 0):
        super().__init__()
        out_dim = dim if out_dim is None else out_dim
        h = expansion * out_dim
        self.dwconv = ChannelsLastConv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.GroupNorm(1, dim, eps=1e-5)
        self.pwconv1 = nn.Linear(dim, h)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(h, out_dim)
        self.skip = nn.Identity() if dim == out_dim else ChannelsLastConv2d(dim, out_dim, 1, bias=False)
        self.pw_dw = None
        if pw_dw:
            self.pw_dw = ChannelsLastConv2d(h, h, pw_dw, padding=pw_dw // 2, groups=h, bias=False)
            with torch.no_grad():
                self.pw_dw.weight.zero_()
                self.pw_dw.weight[:, 0, pw_dw // 2, pw_dw // 2] = 1.0

        self.ada = nn.Linear(cond_dim, 2 * dim) if cond_dim else None

    def forward(self, x, cond=None):
        r = self.skip(x)
        x = self.norm(self.dwconv(x))
        if cond is not None:
            g, b = self.ada(cond).chunk(2, dim=-1)
            x = x * (1 + g[:, :, None, None]) + b[:, :, None, None]
        x = x.permute(0, 2, 3, 1)
        h = self.pwconv1(x)
        if self.pw_dw is not None:
            h = self.pw_dw(h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x = self.pwconv2(self.act(h))
        return r + x.permute(0, 3, 1, 2)


def block_type_of(cfg: Config, name_: str) -> str:
    return getattr(cfg, f"cnn_{name_}_block_type") or cfg.block_type


def _block_factory(
    cfg: Config, cond_dim: int = 0, block_type: str = "", pw_dw: int = 0, v41_expansion: int = 1
):
    if (block_type or cfg.block_type) == "v41":
        return lambda dim, out_dim=None: V41Block(dim, out_dim, cfg.block_norm, cond_dim, v41_expansion)
    return lambda dim, out_dim=None: ConvNeXtBlock(dim, out_dim, cfg.convnext_expansion, pw_dw, cond_dim)


class GridSelfBlock(nn.Module):
    def __init__(self, cfg: Config, dim: int, hw: int):
        super().__init__()
        ls = cfg.layerscale_init
        self.n1, self.attn, self.ls1 = nn.LayerNorm(dim), Attention(cfg, d=dim), LayerScale(dim, ls)
        self.n2, self.ffn, self.ls2 = nn.LayerNorm(dim), FFN(cfg, d=dim), LayerScale(dim, ls)
        cos, sin = rope_table(positions_b(hw, cfg), cfg.head_dim, cfg.rope_base)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, x):
        N, C, H, W = x.shape
        t = x.permute(0, 2, 3, 1).reshape(N, H * W, C)
        h = self.n1(t)
        t = t + self.ls1(self.attn(h, h, rope=(self.cos, self.sin)))
        t = t + self.ls2(self.ffn(self.n2(t)))
        return t.reshape(N, H, W, C).permute(0, 3, 1, 2)


class GridCNN(nn.Module):
    """Convolutional encoder from a one-hot grid to a sequence of tokens."""

    def __init__(
        self,
        cfg: Config,
        stages: tuple,
        depths: tuple,
        downsamples: tuple,
        in_channels: int = N_COLORS,
        cond_dim: int = 0,
        with_attn: bool = False,
        block_type: str = "",
        pw_dw: int = 0,
        v41_expansion: int = 1,
    ):
        super().__init__()
        make_block = _block_factory(cfg, cond_dim, block_type, pw_dw, v41_expansion)
        downs = set(downsamples)
        first_down = min(downs) if (with_attn and downs) else None

        self.entry = ChannelsLastConv2d(in_channels, stages[0], 3, padding=1)
        self.entry_norm = LayerNorm2d(stages[0])

        layers, in_ch = [], stages[0]
        for i, out_ch in enumerate(stages):
            if i == first_down:
                layers.append(GridSelfBlock(cfg, in_ch, GRID_SIZE))
            if i in downs:
                layers.append(
                    nn.Sequential(
                        make_norm(cfg.downsample_norm, in_ch),
                        ChannelsLastConv2d(in_ch, out_ch, 2, stride=2, bias=False),
                    )
                )
                in_ch = out_ch
            for _ in range(depths[i]):
                layers.append(make_block(in_ch))
        self.layers = nn.ModuleList(layers)

        self.out_proj = nn.Linear(in_ch, cfg.d_model)
        self.out_norm = nn.LayerNorm(cfg.d_model)
        self.hw = _hw_final(downsamples)
        self.n_tokens = self.hw * self.hw

    def forward(self, onehot: torch.Tensor, cond=None) -> torch.Tensor:
        x = self.entry_norm(self.entry(onehot))
        for layer in self.layers:
            x = layer(x, cond) if isinstance(layer, (V41Block, ConvNeXtBlock)) else layer(x)

        x = x.permute(0, 2, 3, 1).reshape(x.shape[0], -1, x.shape[1])
        return self.out_norm(self.out_proj(x))


class DemoStem(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        make_block = _block_factory(cfg)
        self.dim = cfg.cnn_in_stages[0] // 2

        self.entry = ChannelsLastConv2d(N_COLORS, self.dim, 3, padding=1)
        self.entry_norm = LayerNorm2d(self.dim)
        self.layers = nn.ModuleList([make_block(self.dim) for _ in range(cfg.cnn_demo_stem_depth)])

    def forward(self, onehot: torch.Tensor) -> torch.Tensor:
        x = self.entry_norm(self.entry(onehot))
        for layer in self.layers:
            x = layer(x)
        return x


def rope_table(positions: list, head_dim: int, base: float, js: tuple = ()) -> tuple:
    """Precomputes the rotary position embedding tables."""
    n_pairs = head_dim // 2
    r = n_pairs // len(positions)
    T = positions[0].shape[0]
    cos = torch.ones(T, n_pairs)
    sin = torch.zeros(T, n_pairs)
    js = js or [range(r)] * len(positions)
    column = 0
    for pos, je in zip(positions, js):
        j = torch.tensor(list(je), dtype=torch.float32)
        inv = base ** (-j / max(r, 1))
        angle = pos.float()[:, None] * inv[None, :]
        cos[:, column : column + len(j)] = torch.cos(angle)
        sin[:, column : column + len(j)] = torch.sin(angle)
        column += len(j)
    return cos, sin


def positions_k(hw: int, scale_: float = 1.0) -> list:
    p = torch.arange(hw * hw)
    off = (scale_ - 1) / 2
    return [(p // hw).float() * scale_ + off, (p % hw).float() * scale_ + off]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applies rotary position embeddings to query or key vectors."""
    B, H, T, D = x.shape
    x = x.view(B, H, T, D // 2, 2)
    x0, x1 = x[..., 0], x[..., 1]
    c = cos[:T].to(x.dtype).view(1, 1, T, D // 2)
    s = sin[:T].to(x.dtype).view(1, 1, T, D // 2)
    return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], dim=-1).view(B, H, T, D)


def demo_cnns(m) -> dict:
    return {n: c for n, c in (("cnn_in", m.cnn_in), ("cnn_out", m.cnn_out)) if c is not None}


def grids_per_demo(cfg: Config) -> int:
    return 1 if cfg.cnn_demo_shared else 2


def positions_demo(max_demos: int, hw: int, sides: tuple = (1, 2)) -> list:
    rows_, cols, grids = [], [], []
    for m in range(max_demos):
        for side in sides:
            for p in range(hw * hw):
                rows_.append(p // hw)
                cols.append(p % hw)
                grids.append(3 * m + side)
    return [torch.tensor(rows_), torch.tensor(cols), torch.tensor(grids)]


def positions_grid(hw: int) -> list:
    p = torch.arange(hw * hw)
    return [p // hw, p % hw]


def positions_b(hw: int, cfg: Config) -> list:
    pb = positions_grid(hw)
    return with_diagonals(pb) if cfg.rope_diag else pb


def with_diagonals(pos: list) -> list:
    f, c, *rest = pos
    return [f, c, *rest, f + c, f - c]


class Attention(nn.Module):
    """Multi-head attention with rotary position embeddings."""

    def __init__(self, cfg: Config, d: int = None):
        super().__init__()
        d_mod = cfg.d_model if d is None else d
        self.h = cfg.n_heads if d is None else d // cfg.head_dim
        self.dh = cfg.head_dim
        d_int = self.h * self.dh
        self.q = nn.Linear(d_mod, d_int)
        self.k = nn.Linear(d_mod, d_int)
        self.v = nn.Linear(d_mod, d_int)
        self.o = nn.Linear(d_int, d_mod)
        self.p = cfg.dropout

        self.gate = cfg.attn_gate
        if self.gate:
            self.g = nn.Linear(d_mod, d_int if self.gate == "elementwise" else self.h, bias=False)

    def _heads(self, t, B, L):
        return t.view(B, L, self.h, self.dh).transpose(1, 2)

    def forward(self, xq, xkv, rope=None, keep_kv=None, rope_kv=None, rope_q=None):
        B, Lq, _ = xq.shape
        Lk = xkv.shape[1]
        q = self._heads(self.q(xq), B, Lq)
        k = self._heads(self.k(xkv), B, Lk)
        v = self._heads(self.v(xkv), B, Lk)
        if rope is not None:
            cos, sin = rope
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        if rope_kv is not None:
            k = apply_rope(k, *rope_kv)
        if rope_q is not None:
            q = apply_rope(q, *rope_q)

        mask = None if keep_kv is None else keep_kv[:, None, None, :]
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.p if self.training else 0.0
        )
        x = x.transpose(1, 2)
        if self.gate:
            x = x * torch.sigmoid(self.g(xq)).view(B, Lq, self.h, -1)
        return self.o(x.reshape(B, Lq, self.h * self.dh))


class FFN(nn.Module):
    """Feed-forward block."""

    def __init__(self, cfg: Config, d: int = None, dw: int = 0):
        super().__init__()
        d = cfg.d_model if d is None else d
        h = cfg.ffn_mult * d
        self.fc1 = nn.Linear(d, h)
        self.fc2 = nn.Linear(h, d)
        self.drop = nn.Dropout(cfg.dropout)
        self.dw = None
        if dw:
            self.dw = ChannelsLastConv2d(h, h, dw, padding=dw // 2, groups=h, bias=False)
            with torch.no_grad():
                self.dw.weight.zero_()
                self.dw.weight[:, 0, dw // 2, dw // 2] = 1.0

    def forward(self, x):
        h = self.fc1(x)
        if self.dw is not None:
            B, L, C = h.shape
            s = int(round(L**0.5))
            h = self.dw(h.view(B, s, s, C).permute(0, 3, 1, 2))
            h = h.permute(0, 2, 3, 1).reshape(B, L, C)
        return self.drop(self.fc2(F.gelu(h)))


class UnfoldPatches(nn.Module):
    def __init__(self, f: int):
        super().__init__()
        self.f = f

    def forward(self, x):
        B, L, _ = x.shape
        hw, f = int(round(L**0.5)), self.f
        x = x.view(B, hw, hw, f, f, N_COLORS).permute(0, 1, 3, 2, 4, 5)
        return x.reshape(B, hw * f * hw * f, N_COLORS)


class LayerScale(nn.Module):
    """Per-channel learnable scaling of a residual branch."""

    def __init__(self, d_model: int, init: float):
        super().__init__()
        self.gamma = nn.Parameter(init * torch.ones(d_model))

    def forward(self, x):
        return self.gamma * x


def _as_source(v: torch.Tensor) -> torch.Tensor:
    if torch.is_autocast_enabled(v.device.type):
        return v.to(torch.get_autocast_dtype(v.device.type))
    return v


def stream_new(v: torch.Tensor) -> list:
    return [_as_source(v)]


def stream_append(stream: list, v: torch.Tensor) -> list:
    return stream + [_as_source(v)]


class _DepthReadFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, *vs):
        qf, vf = q.float(), [v.float() for v in vs]
        r = torch.stack([torch.rsqrt(v.square().mean(-1) + 1e-6) for v in vf])
        z = torch.stack([(v * qf).sum(-1) for v in vf])
        a = (z * r).softmax(0)
        h = a[0, ..., None] * vf[0]
        for i in range(1, len(vf)):
            h = h + a[i, ..., None] * vf[i]
        ctx.save_for_backward(q, a, r, z, *vs)
        return h

    @staticmethod
    def backward(ctx, dh):
        q, a, r, z, *vs = ctx.saved_tensors
        dh, qf, vf = dh.float(), q.float(), [v.float() for v in vs]
        d = vf[0].shape[-1]
        da = torch.stack([(dh * v).sum(-1) for v in vf])
        dl = a * (da - (a * da).sum(0, keepdim=True))
        dz = dl * r
        dr3 = dl * z * r.pow(3) / d
        dvs = [
            (a[i, ..., None] * dh + dz[i, ..., None] * qf - dr3[i, ..., None] * vf[i]).to(vs[i].dtype)
            for i in range(len(vf))
        ]
        dq = dz[0, ..., None] * vf[0]
        for i in range(1, len(vf)):
            dq = dq + dz[i, ..., None] * vf[i]
        if q.dim() == 1:
            dq = dq.reshape(-1, d).sum(0)
        return (dq.to(q.dtype), *dvs)


try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

if triton is not None:

    @triton.jit
    def _ar_fwd(P, Q, H, R, Z, LSE, T, n, sq, D: tl.constexpr, BT: tl.constexpr, BF16: tl.constexpr):
        rows = tl.program_id(0) * BT + tl.arange(0, BT)
        rm = rows < T
        cols = tl.arange(0, D)
        off = rows[:, None] * D + cols[None, :]
        q = tl.load(Q + rows[:, None] * sq + cols[None, :], mask=rm[:, None], other=0.0)
        m = tl.full([BT], float("-inf"), tl.float32)
        s = tl.zeros([BT], tl.float32)
        acc = tl.zeros([BT, D], tl.float32)
        for i in range(n):
            if BF16:
                p = tl.load(P + i).to(tl.pointer_type(tl.bfloat16))
            else:
                p = tl.load(P + i).to(tl.pointer_type(tl.float32))
            v = tl.load(p + off, mask=rm[:, None], other=0.0).to(tl.float32)
            r = tl.rsqrt(tl.sum(v * v, 1) / D + 1e-6)
            z = tl.sum(v * q, 1)
            tl.store(R + i * T + rows, r, mask=rm)
            tl.store(Z + i * T + rows, z, mask=rm)
            l = z * r
            mn = tl.maximum(m, l)
            c = tl.exp(m - mn)
            e = tl.exp(l - mn)
            s = s * c + e
            acc = acc * c[:, None] + e[:, None] * v
            m = mn
        tl.store(H + off, acc / s[:, None], mask=rm[:, None])
        tl.store(LSE + rows, m + tl.log(s), mask=rm)

    @triton.jit
    def _ar_bwd(
        P,
        G,
        Q,
        DH,
        R,
        Z,
        LSE,
        DQ,
        T,
        n,
        sq,
        D: tl.constexpr,
        BT: tl.constexpr,
        BF16: tl.constexpr,
        QTOK: tl.constexpr,
    ):
        rows = tl.program_id(0) * BT + tl.arange(0, BT)
        rm = rows < T
        cols = tl.arange(0, D)
        off = rows[:, None] * D + cols[None, :]
        q = tl.load(Q + rows[:, None] * sq + cols[None, :], mask=rm[:, None], other=0.0)
        dh = tl.load(DH + off, mask=rm[:, None], other=0.0)
        lse = tl.load(LSE + rows, mask=rm, other=0.0)
        S = tl.zeros([BT], tl.float32)
        for i in range(n):
            if BF16:
                p = tl.load(P + i).to(tl.pointer_type(tl.bfloat16))
            else:
                p = tl.load(P + i).to(tl.pointer_type(tl.float32))
            v = tl.load(p + off, mask=rm[:, None], other=0.0).to(tl.float32)
            r = tl.load(R + i * T + rows, mask=rm, other=0.0)
            z = tl.load(Z + i * T + rows, mask=rm, other=0.0)
            S += tl.exp(z * r - lse) * tl.sum(dh * v, 1)
        dq = tl.zeros([BT, D], tl.float32)
        for i in range(n):
            if BF16:
                p = tl.load(P + i).to(tl.pointer_type(tl.bfloat16))
                g = tl.load(G + i).to(tl.pointer_type(tl.bfloat16))
            else:
                p = tl.load(P + i).to(tl.pointer_type(tl.float32))
                g = tl.load(G + i).to(tl.pointer_type(tl.float32))
            v = tl.load(p + off, mask=rm[:, None], other=0.0).to(tl.float32)
            r = tl.load(R + i * T + rows, mask=rm, other=0.0)
            z = tl.load(Z + i * T + rows, mask=rm, other=0.0)
            a = tl.exp(z * r - lse)
            dl = a * (tl.sum(dh * v, 1) - S)
            dz = dl * r
            dr3 = dl * z * r * r * r / D
            dv = a[:, None] * dh + dz[:, None] * q - dr3[:, None] * v
            tl.store(g + off, dv.to(g.dtype.element_ty), mask=rm[:, None])
            dq += dz[:, None] * v
        if QTOK:
            tl.store(DQ + off, dq, mask=rm[:, None])
        else:
            tl.atomic_add(DQ + cols, tl.sum(dq, 0))

    _AR_CFG = {"fwd": (8, 4), "bwd": (4, 4)}
    _AR_NS = f"cailar{time.time_ns() % 10**9}"

    def _pointer_table(ts: list) -> torch.Tensor:
        t = torch.tensor([x.data_ptr() for x in ts], dtype=torch.int64)
        return t.pin_memory().to(ts[0].device, non_blocking=True)

    def _flat_query(q: torch.Tensor, d: int) -> tuple:
        qf = q.float().contiguous()
        return qf, (0 if qf.dim() == 1 else d)

    @torch.library.custom_op(f"{_AR_NS}::fwd", mutates_args=())
    def _ar_fwd_op(
        q: torch.Tensor, vs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        vs = [v.contiguous() for v in vs]
        shape, d = vs[0].shape, vs[0].shape[-1]
        T, n = vs[0].numel() // d, len(vs)
        qf, sq = _flat_query(q, d)
        dev = vs[0].device
        h = torch.empty(shape, device=dev, dtype=torch.float32)
        r = torch.empty(n, T, device=dev, dtype=torch.float32)
        z = torch.empty(n, T, device=dev, dtype=torch.float32)
        lse = torch.empty(T, device=dev, dtype=torch.float32)
        bt, nw = _AR_CFG["fwd"]
        _ar_fwd[(triton.cdiv(T, bt),)](
            _pointer_table(vs),
            qf,
            h,
            r,
            z,
            lse,
            T,
            n,
            sq,
            D=d,
            BT=bt,
            BF16=vs[0].dtype == torch.bfloat16,
            num_warps=nw,
        )
        return h, r, z, lse

    @_ar_fwd_op.register_fake
    def _(q, vs):
        d = vs[0].shape[-1]
        T, n = vs[0].numel() // d, len(vs)
        e = vs[0].new_empty
        return (
            e(vs[0].shape, dtype=torch.float32),
            e((n, T), dtype=torch.float32),
            e((n, T), dtype=torch.float32),
            e((T,), dtype=torch.float32),
        )

    @torch.library.custom_op(f"{_AR_NS}::bwd", mutates_args=())
    def _ar_bwd_op(
        dh: torch.Tensor,
        q: torch.Tensor,
        vs: list[torch.Tensor],
        r: torch.Tensor,
        z: torch.Tensor,
        lse: torch.Tensor,
    ) -> list[torch.Tensor]:
        vs = [v.contiguous() for v in vs]
        d = vs[0].shape[-1]
        T, n = vs[0].numel() // d, len(vs)
        qf, sq = _flat_query(q, d)
        dvs = [torch.empty_like(v) for v in vs]
        dq = torch.zeros_like(qf)
        bt, nw = _AR_CFG["bwd"]
        _ar_bwd[(triton.cdiv(T, bt),)](
            _pointer_table(vs),
            _pointer_table(dvs),
            qf,
            dh.float().contiguous(),
            r,
            z,
            lse,
            dq,
            T,
            n,
            sq,
            D=d,
            BT=bt,
            BF16=vs[0].dtype == torch.bfloat16,
            QTOK=sq != 0,
            num_warps=nw,
        )
        return [dq.to(q.dtype), *dvs]

    @_ar_bwd_op.register_fake
    def _(dh, q, vs, r, z, lse):
        return [q.new_empty(q.shape), *[v.new_empty(v.shape) for v in vs]]

    def _ar_setup(ctx, inputs, output):
        q, vs = inputs
        _, r, z, lse = output
        ctx.n = len(vs)
        ctx.save_for_backward(q, r, z, lse, *vs)

    def _ar_backward(ctx, dh, _dr, _dz, _dlse):
        q, r, z, lse, *vs = ctx.saved_tensors
        dq, *dvs = _ar_bwd_op(dh, q, vs, r, z, lse)
        return dq, dvs

    _ar_fwd_op.register_autograd(_ar_backward, setup_context=_ar_setup)


def read_sources(q: torch.Tensor, vs: list) -> torch.Tensor:
    """Computes the Attention Residuals weighted sum of the sources."""
    if triton is not None and vs[0].is_cuda:
        return _ar_fwd_op(q, vs)[0]
    return _DepthReadFn.apply(q, *vs)


class DepthReader(nn.Module):
    """One Attention Residuals read: a softmax over the sources of a stream."""

    def __init__(self, d: int, projected: bool, from_: str = "sum"):
        super().__init__()
        self.projected = projected
        self.from_ = from_
        if projected:
            self.project = nn.Linear(d, d)
        else:
            self.w = nn.Parameter(torch.zeros(d))

    def query(self, vs: list) -> torch.Tensor:
        if not self.projected:
            return self.w
        if self.from_ == "last":
            s = vs[-1].float()
        else:
            s = vs[0].float()
            for v in vs[1:]:
                s = s + v.float()
        return self.project(F.rms_norm(s, (s.shape[-1],), eps=1e-6))

    def forward(self, vs: list) -> torch.Tensor:

        q = self.query(vs)

        return read_sources(q, vs)


def depth_reader(cfg: Config) -> DepthReader:
    return DepthReader(cfg.d_model, cfg.attn_res_projected_query, cfg.attn_res_query_from)


class SelfBlock(nn.Module):
    """Pre-norm self-attention block."""

    def __init__(self, cfg: Config, px: bool = False):
        super().__init__()
        dw = cfg.ffn_px_dw if px else 0
        self.n1, self.attn, self.ls1 = (
            nn.LayerNorm(cfg.d_model),
            Attention(cfg),
            LayerScale(cfg.d_model, cfg.layerscale_init),
        )
        self.n2, self.ffn, self.ls2 = (
            nn.LayerNorm(cfg.d_model),
            FFN(cfg, dw=dw),
            LayerScale(cfg.d_model, cfg.layerscale_init),
        )
        self.depth_mode = cfg.attn_res
        if self.depth_mode:
            self.r1, self.r2 = depth_reader(cfg), depth_reader(cfg)

    def forward(self, x, rope=None, keep=None):
        if self.depth_mode:
            h = self.n1(self.r1(x))
            x = stream_append(x, self.ls1(self.attn(h, h, rope=rope, keep_kv=keep)))
            return stream_append(x, self.ls2(self.ffn(self.n2(self.r2(x)))))
        h = self.n1(x)
        x = x + self.ls1(self.attn(h, h, rope=rope, keep_kv=keep))
        return x + self.ls2(self.ffn(self.n2(x)))


class RuleBlock(nn.Module):
    """Cross-attention block for the bottleneck tokens variant."""

    def __init__(self, cfg: Config):
        super().__init__()
        d, ls = cfg.d_model, cfg.layerscale_init
        self.nq, self.nkv, self.cross, self.ls1 = (
            nn.LayerNorm(d),
            nn.LayerNorm(d),
            Attention(cfg),
            LayerScale(d, ls),
        )
        self.n2, self.self_attn, self.ls2 = nn.LayerNorm(d), Attention(cfg), LayerScale(d, ls)
        self.n3, self.ffn, self.ls3 = nn.LayerNorm(d), FFN(cfg), LayerScale(d, ls)
        self.depth_mode = cfg.attn_res
        if self.depth_mode:
            self.r_q, self.r_kv = depth_reader(cfg), depth_reader(cfg)
            self.r2, self.r3 = depth_reader(cfg), depth_reader(cfg)

    def forward(self, k, ctx, keep_ctx=None, rope_ctx=None, rope_k=None, rope_self=None, keep_self=None):
        if self.depth_mode:
            k = stream_append(
                k,
                self.ls1(
                    self.cross(
                        self.nq(self.r_q(k)),
                        self.nkv(self.r_kv(ctx)),
                        keep_kv=keep_ctx,
                        rope_kv=rope_ctx,
                        rope_q=rope_k,
                    )
                ),
            )
            h = self.n2(self.r2(k))
            k = stream_append(k, self.ls2(self.self_attn(h, h, rope=rope_self, keep_kv=keep_self)))
            return stream_append(k, self.ls3(self.ffn(self.n3(self.r3(k)))))
        k = k + self.ls1(
            self.cross(self.nq(k), self.nkv(ctx), keep_kv=keep_ctx, rope_kv=rope_ctx, rope_q=rope_k)
        )
        h = self.n2(k)
        k = k + self.ls2(self.self_attn(h, h, rope=rope_self, keep_kv=keep_self))
        return k + self.ls3(self.ffn(self.n3(k)))


class DecoderBlock(nn.Module):
    """One-way decoder block for the bottleneck tokens variant."""

    def __init__(self, cfg: Config):
        super().__init__()
        d, ls = cfg.d_model, cfg.layerscale_init

        self.with_self, self.with_cross_k = cfg.b4_self, cfg.b4_cross_k
        if self.with_self:
            self.n1, self.self_attn, self.ls1 = nn.LayerNorm(d), Attention(cfg), LayerScale(d, ls)
        if self.with_cross_k:
            self.nk_q, self.nk_kv, self.cross_k, self.ls_k = (
                nn.LayerNorm(d),
                nn.LayerNorm(d),
                Attention(cfg),
                LayerScale(d, ls),
            )
        self.nq, self.nkv, self.cross, self.ls2 = (
            nn.LayerNorm(d),
            nn.LayerNorm(d),
            Attention(cfg),
            LayerScale(d, ls),
        )
        self.n3, self.ffn, self.ls3 = nn.LayerNorm(d), FFN(cfg), LayerScale(d, ls)

        self.depth_mode = cfg.attn_res
        if self.depth_mode:
            if self.with_self:
                self.r1 = depth_reader(cfg)
            self.r_q, self.r_kv, self.r3 = depth_reader(cfg), depth_reader(cfg), depth_reader(cfg)

    def forward(self, x, k, rope=None, rope_q=None, rope_k=None, rope_px=None, rope_kp=None, keep_k=None):
        if self.depth_mode:
            if self.with_self:
                h = self.n1(self.r1(x))
                x = stream_append(x, self.ls1(self.self_attn(h, h, rope=rope)))
            x = stream_append(
                x,
                self.ls2(
                    self.cross(
                        self.nq(self.r_q(x)),
                        self.nkv(self.r_kv(k)),
                        rope_q=rope_q,
                        rope_kv=rope_k,
                        keep_kv=keep_k,
                    )
                ),
            )
            return stream_append(x, self.ls3(self.ffn(self.n3(self.r3(x))))), k
        if self.with_self:
            h = self.n1(x)
            x = x + self.ls1(self.self_attn(h, h, rope=rope))
        if self.with_cross_k:
            k = k + self.ls_k(self.cross_k(self.nk_q(k), self.nk_kv(x), rope_kv=rope_px, rope_q=rope_kp))
        x = x + self.ls2(self.cross(self.nq(x), self.nkv(k), rope_q=rope_q, rope_kv=rope_k, keep_kv=keep_k))
        return x + self.ls3(self.ffn(self.n3(x))), k


class TwoWayBlock(nn.Module):
    """Two-way decoder block: demonstrations read the pixels, then the pixels read the demonstrations."""

    def __init__(self, cfg: Config, with_pixel_self: bool = False, with_pixel_attn: bool = False):
        super().__init__()
        d, ls = cfg.d_model, cfg.layerscale_init
        self.reads_test = cfg.demos_read_test
        if self.reads_test:
            self.nd_q, self.nd_kv, self.cross_d, self.ls_d1 = (
                nn.LayerNorm(d),
                nn.LayerNorm(d),
                Attention(cfg),
                LayerScale(d, ls),
            )
        self.nd_s, self.self_d, self.ls_d2 = nn.LayerNorm(d), Attention(cfg), LayerScale(d, ls)
        self.nd_f, self.ffn_d, self.ls_d3 = nn.LayerNorm(d), FFN(cfg), LayerScale(d, ls)
        self.with_pixel_self = with_pixel_self
        if with_pixel_self:
            self.np_s, self.self_p, self.ls_p0 = nn.LayerNorm(d), Attention(cfg), LayerScale(d, ls)
        self.np_q, self.np_kv, self.cross_p, self.ls_p1 = (
            nn.LayerNorm(d),
            nn.LayerNorm(d),
            Attention(cfg),
            LayerScale(d, ls),
        )
        self.with_pixel_attn = with_pixel_attn
        if with_pixel_attn:
            self.np_a, self.attn_p, self.ls_pa = nn.LayerNorm(d), Attention(cfg), LayerScale(d, ls)
        self.np_f, self.ffn_p, self.ls_p2 = nn.LayerNorm(d), FFN(cfg, dw=cfg.ffn_px_dw), LayerScale(d, ls)
        self.depth_mode = cfg.attn_res
        if self.depth_mode:
            if self.reads_test:
                self.r_d1, self.r_d1kv = depth_reader(cfg), depth_reader(cfg)
            self.r_d2, self.r_d3 = depth_reader(cfg), depth_reader(cfg)
            if with_pixel_self:
                self.r_p0 = depth_reader(cfg)
            self.r_p1, self.r_p1kv = depth_reader(cfg), depth_reader(cfg)
            if with_pixel_attn:
                self.r_pa = depth_reader(cfg)
            self.r_p2 = depth_reader(cfg)

    def forward(self, x, seq, keep, rope_a=None, rope_b=None, rope_px_k=None, rope_px_q=None):
        if self.depth_mode:
            return self._forward_depth_block(x, seq, keep, rope_a, rope_b, rope_px_k, rope_px_q)
        if self.reads_test:
            seq = seq + self.ls_d1(
                self.cross_d(self.nd_q(seq), self.nd_kv(x), rope_q=rope_a, rope_kv=rope_px_k)
            )
        h = self.nd_s(seq)
        seq = seq + self.ls_d2(self.self_d(h, h, rope=rope_a, keep_kv=keep))
        seq = seq + self.ls_d3(self.ffn_d(self.nd_f(seq)))
        if self.with_pixel_self:
            h = self.np_s(x)
            x = x + self.ls_p0(self.self_p(h, h, rope=rope_b))
        x = x + self.ls_p1(
            self.cross_p(self.np_q(x), self.np_kv(seq), rope_q=rope_px_q, rope_kv=rope_a, keep_kv=keep)
        )
        if self.with_pixel_attn:
            h = self.np_a(x)
            x = x + self.ls_pa(self.attn_p(h, h, rope=rope_b))
        return x + self.ls_p2(self.ffn_p(self.np_f(x))), seq

    def _forward_depth_block(self, x, seq, keep, rope_a, rope_b, rope_px_k, rope_px_q):
        if self.reads_test:
            seq = stream_append(
                seq,
                self.ls_d1(
                    self.cross_d(
                        self.nd_q(self.r_d1(seq)),
                        self.nd_kv(self.r_d1kv(x)),
                        rope_q=rope_a,
                        rope_kv=rope_px_k,
                    )
                ),
            )
        h = self.nd_s(self.r_d2(seq))
        seq = stream_append(seq, self.ls_d2(self.self_d(h, h, rope=rope_a, keep_kv=keep)))
        seq = stream_append(seq, self.ls_d3(self.ffn_d(self.nd_f(self.r_d3(seq)))))
        if self.with_pixel_self:
            h = self.np_s(self.r_p0(x))
            x = stream_append(x, self.ls_p0(self.self_p(h, h, rope=rope_b)))
        x = stream_append(
            x,
            self.ls_p1(
                self.cross_p(
                    self.np_q(self.r_p1(x)),
                    self.np_kv(self.r_p1kv(seq)),
                    rope_q=rope_px_q,
                    rope_kv=rope_a,
                    keep_kv=keep,
                )
            ),
        )
        if self.with_pixel_attn:
            h = self.np_a(self.r_pa(x))
            x = stream_append(x, self.ls_pa(self.attn_p(h, h, rope=rope_b)))
        return stream_append(x, self.ls_p2(self.ffn_p(self.np_f(self.r_p2(x))))), seq


class TaskConditioning(nn.Module):
    """Learned query that summarizes the demonstrations into the task vector."""

    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_model
        self.mode = cfg.cond_test
        self.norm = nn.LayerNorm(d)
        if self.mode == "cross":
            self.query = nn.Parameter(torch.zeros(1, 1, d))
            nn.init.trunc_normal_(self.query, std=0.02)
            self.attn = Attention(cfg)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d), nn.SiLU())

    def forward(self, seq, keep):
        h = self.norm(seq)
        if self.mode == "mean":
            w = keep.to(h.dtype)[..., None]
            v = (h * w).sum(1) / w.sum(1)
        else:
            q = self.query.expand(seq.shape[0], -1, -1)
            v = self.attn(q, h, keep_kv=keep)[:, 0]
        return self.mlp(v)


CNN_BATCH_MULTIPLE = 64


def _round_up(n: int, multiple: int) -> int:
    return ((n + multiple - 1) // multiple) * multiple


def one_hot_grids(u8: torch.Tensor, n_pad: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """One-hot encodes grids, padding the batch with zeros up to n_pad."""
    n = u8.shape[0]
    out = (
        torch.empty((n_pad, N_COLORS, GRID_SIZE, GRID_SIZE), dtype=dtype, device=u8.device)
        .to(memory_format=torch.channels_last)
        .zero_()
    )
    out[:n].scatter_(1, u8[:, None].long(), 1.0)
    return out


class CAIL(nn.Module):
    """The Change-Aware In-Context Learner."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.stem_demo = DemoStem(cfg) if cfg.cnn_demo_stem else None
        demo_channels = (
            2 * self.stem_demo.dim
            if cfg.cnn_demo_stem
            else 2 * N_COLORS
            if cfg.cnn_demo_sees_pair
            else N_COLORS
        )
        if cfg.cnn_demo_sees_test:
            demo_channels += N_COLORS
        self.cnn_in = GridCNN(
            cfg,
            cfg.cnn_in_stages,
            cfg.cnn_in_depths,
            cfg.cnn_in_downsamples,
            demo_channels,
            with_attn=cfg.cnn_demo_attn,
            block_type=block_type_of(cfg, "in"),
            pw_dw=cfg.cnn_demo_pw_dw,
        )
        self.cnn_out = (
            None
            if cfg.cnn_demo_shared
            else GridCNN(
                cfg,
                cfg.cnn_out_stages,
                cfg.cnn_out_depths,
                cfg.cnn_out_downsamples,
                demo_channels,
                with_attn=cfg.cnn_demo_attn,
                block_type=block_type_of(cfg, "out"),
                pw_dw=cfg.cnn_demo_pw_dw,
            )
        )
        self.cnn_test = GridCNN(
            cfg,
            cfg.cnn_test_stages,
            cfg.cnn_test_depths,
            cfg.cnn_test_downsamples,
            cond_dim=cfg.d_model if cfg.cond_test else 0,
            block_type=block_type_of(cfg, "test"),
            v41_expansion=cfg.cnn_test_v41_expansion,
        )
        self.cond_test = TaskConditioning(cfg) if cfg.cond_test else None

        with_grid = cfg.grid_cross_before != -1
        self.cnn_grid = (
            GridCNN(
                cfg,
                cfg.cnn_grid_stages,
                cfg.cnn_grid_depths,
                cfg.cnn_grid_downsamples,
                block_type=block_type_of(cfg, "grid"),
            )
            if with_grid
            else None
        )
        self.grid_cross = RuleBlock(cfg) if with_grid else None

        self.a2 = nn.ModuleList([SelfBlock(cfg) for _ in range(cfg.n_blocks_a2)])
        self.a3 = nn.ModuleList([RuleBlock(cfg) for _ in range(cfg.n_blocks_a3)])
        self.b2 = nn.ModuleList([SelfBlock(cfg, px=True) for _ in range(cfg.n_blocks_b2)])
        self.b3 = nn.ModuleList([RuleBlock(cfg) for _ in range(cfg.n_blocks_b3)])

        if cfg.decoder == "two_way":
            every, n = cfg.pixel_self_every, cfg.n_blocks_b4
            self.b4 = nn.ModuleList(
                [
                    TwoWayBlock(
                        cfg,
                        with_pixel_self=bool(every) and i % every == 0,
                        with_pixel_attn=cfg.b4_pixelattn and (n - 1 - i) % cfg.b4_pixelattn_every == 0,
                    )
                    for i in range(n)
                ]
            )
        else:
            self.b4 = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_blocks_b4)])

        self.no_bottleneck = cfg.bridge == "demos"
        if self.no_bottleneck:
            self.bottleneck_tokens = None
        else:
            self.bottleneck_tokens = nn.Parameter(torch.zeros(cfg.n_bottleneck_tokens, cfg.d_model))
            nn.init.trunc_normal_(self.bottleneck_tokens, std=0.02)

        f = GRID_SIZE // self.cnn_test.hw
        self.head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, N_COLORS * f * f),
            *([UnfoldPatches(f)] if f > 1 else []),
        )

        if cfg.attn_res:
            self.reader_cond = depth_reader(cfg) if cfg.cond_test else None
            self.reader_head = depth_reader(cfg)

        pa = positions_demo(cfg.max_demos, self.cnn_in.hw, (1, 2)[: grids_per_demo(cfg)])
        if cfg.rope_diag:
            pa = with_diagonals(pa)
        cos_a, sin_a = rope_table(pa, cfg.head_dim, cfg.rope_base, cfg.rope_a_js)
        self.register_buffer("cos_a", cos_a, persistent=False)
        self.register_buffer("sin_a", sin_a, persistent=False)

        if with_grid:
            pg = positions_demo(cfg.max_demos, self.cnn_grid.hw, (1, 2))
            if cfg.rope_diag:
                pg = with_diagonals(pg)
            cos_g, sin_g = rope_table(pg, cfg.head_dim, cfg.rope_base, cfg.rope_a_js)
            self.register_buffer("cos_g", cos_g, persistent=False)
            self.register_buffer("sin_g", sin_g, persistent=False)
        pb = positions_b(self.cnn_test.hw, cfg)
        cos_b, sin_b = rope_table(pb, cfg.head_dim, cfg.rope_base)
        self.register_buffer("cos_b", cos_b, persistent=False)
        self.register_buffer("sin_b", sin_b, persistent=False)
        cos_b4, sin_b4 = rope_table(pb, cfg.head_dim, cfg.rope_base, cfg.rope_b4_js)
        self.register_buffer("cos_b4", cos_b4, persistent=False)
        self.register_buffer("sin_b4", sin_b4, persistent=False)

        hw_k = self.cnn_in.hw
        fk, ck = positions_k(hw_k)
        pka = [fk, ck, torch.zeros_like(fk)]
        if cfg.rope_diag:
            pka = with_diagonals(pka)
        cos_ka, sin_ka = rope_table(pka, cfg.head_dim, cfg.rope_base, cfg.rope_a_js)
        self.register_buffer("cos_ka", cos_ka, persistent=False)
        self.register_buffer("sin_ka", sin_ka, persistent=False)
        pkb = positions_k(hw_k, self.cnn_test.hw / hw_k)
        if cfg.rope_diag:
            pkb = with_diagonals(pkb)
        cos_kb, sin_kb = rope_table(pkb, cfg.head_dim, cfg.rope_base, cfg.rope_b4_js)
        self.register_buffer("cos_kb", cos_kb, persistent=False)
        self.register_buffer("sin_kb", sin_kb, persistent=False)

        self.register_buffer("frame", FRAME.clone(), persistent=False)
        frame_logits = torch.full((N_COLORS,), -1e4)
        frame_logits[EMPTY_VALUE] = 0.0
        self.register_buffer("frame_logits", frame_logits, persistent=False)

        self.apply(self._init)

        self.previous_logits = []

        for m in self.cnn_test.modules():
            if isinstance(m, (V41Block, ConvNeXtBlock)) and m.ada is not None:
                nn.init.zeros_(m.ada.weight)
                nn.init.zeros_(m.ada.bias)

        for m in self.modules():
            if isinstance(m, DepthReader) and m.projected:
                nn.init.zeros_(m.project.weight)

        for c in self._convs():
            c.to(memory_format=torch.channels_last)

    def _decode(self, x):
        lg = self.head(x)
        if self.cfg.free_frame:
            lg = torch.where(self.frame[None, :, None], self.frame_logits.to(lg.dtype), lg)
        return lg

    def _convs(self) -> tuple:
        return tuple(
            c
            for c in (self.stem_demo, self.cnn_in, self.cnn_out, self.cnn_grid, self.cnn_test)
            if c is not None
        )

    def compile_model(self):

        torch._dynamo.config.recompile_limit = 32
        for c in self._convs():
            c.forward = torch.compile(c.forward, dynamic=False)
        for stack_ in (self.a2, self.a3, self.b2, self.b3, self.b4):
            for blk in stack_:
                blk.forward = torch.compile(blk.forward, dynamic=False)
        for m in (self.cond_test, self.grid_cross):
            if m is not None:
                m.forward = torch.compile(m.forward, dynamic=False)
        self.head.forward = torch.compile(self.head.forward, dynamic=False)
        for reader in (getattr(self, "reader_cond", None), getattr(self, "reader_head", None)):
            if reader is not None:
                reader.forward = torch.compile(reader.forward, dynamic=False)
        return self

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _pack_demos(self, tok_in, tok_out, n_demos: list):
        toks = [tok_in] if tok_out is None else [tok_in, tok_out]
        G = len(toks)
        B_, P, d = len(n_demos), tok_in.shape[1], tok_in.shape[2]
        max_d = max(n_demos)
        nd = np.asarray(n_demos)
        S = int(nd.sum())

        task_ = np.repeat(np.arange(B_), nd)
        inside = np.arange(S) - np.repeat(np.cumsum(nd) - nd, nd)
        base = task_ * (G * max_d) + G * inside

        dest = (base[:, None] + np.arange(G)).reshape(-1).astype(np.int64)
        dest = torch.from_numpy(dest).to(tok_in.device, non_blocking=True)

        pair = torch.stack(toks, dim=1).reshape(-1, P, d)
        seq = tok_in.new_zeros((B_ * G * max_d, P, d))
        seq.index_copy_(0, dest, pair)
        keep = torch.zeros(B_ * G * max_d, dtype=torch.bool, device=tok_in.device)
        keep.index_fill_(0, dest, True)

        return (
            seq.reshape(B_, G * max_d * P, d),
            keep.reshape(B_, G * max_d, 1).expand(B_, G * max_d, P).reshape(B_, -1),
        )

    def forward(self, demo_in_u8, demo_out_u8, test_in_u8, n_demos: list):
        dev = test_in_u8.device
        S = demo_in_u8.shape[0]
        S_pad = _round_up(S, CNN_BATCH_MULTIPLE)

        dt = torch.get_autocast_dtype(dev.type) if torch.is_autocast_enabled() else torch.float32

        oh_in = one_hot_grids(demo_in_u8, S_pad, dt)
        oh_out = one_hot_grids(demo_out_u8, S_pad, dt)

        grid = None
        if self.cnn_grid is not None:
            g_in, g_out = self.cnn_grid(
                torch.cat([oh_in, oh_out], dim=0).contiguous(memory_format=torch.channels_last)
            ).chunk(2, dim=0)
            grid = self._pack_demos(g_in[:S], g_out[:S], n_demos)
        if self.stem_demo is not None:
            oh_in, oh_out = self.stem_demo(torch.cat([oh_in, oh_out], dim=0)).chunk(2, dim=0)
        if self.cfg.cnn_demo_sees_pair:
            grids = [oh_in, oh_out]
            if self.cfg.cnn_demo_sees_test:
                task_ = torch.repeat_interleave(
                    torch.arange(len(n_demos), device=dev), torch.as_tensor(n_demos, device=dev)
                )
                grids.append(one_hot_grids(test_in_u8[task_], S_pad, dt))
            pair = torch.cat(grids, dim=1).contiguous(memory_format=torch.channels_last)
            oh_in = oh_out = pair
        tok_in = self.cnn_in(oh_in)[:S]
        tok_out = None if self.cnn_out is None else self.cnn_out(oh_out)[:S]

        seq, keep = self._pack_demos(tok_in, tok_out, n_demos)
        rope_a = (self.cos_a[: seq.shape[1]], self.sin_a[: seq.shape[1]])
        if self.cfg.attn_res:
            return self._forward_depth(seq, keep, rope_a, test_in_u8, dt)
        for blk in self.a2:
            seq = blk(seq, rope=rope_a, keep=keep)

        cond = self.cond_test(seq, keep) if self.cond_test is not None else None
        x = self.cnn_test(one_hot_grids(test_in_u8, test_in_u8.shape[0], dt), cond)
        rope_b = (self.cos_b, self.sin_b)
        for blk in self.b2:
            x = blk(x, rope=rope_b)

        if self.no_bottleneck:
            return self._decode_no_bottleneck(seq, keep, rope_a, x, rope_b, grid)

        k = self.bottleneck_tokens.unsqueeze(0).expand(len(n_demos), -1, -1)
        rope_ctx = rope_a if self.cfg.rope_a3_k else None
        rope_ka = (self.cos_ka, self.sin_ka) if self.cfg.rope_k_a3_q else None
        rope_kk = (self.cos_ka, self.sin_ka) if self.cfg.rope_k_self else None
        rope_px = (self.cos_b4, self.sin_b4) if self.cfg.rope_b3 else None
        rope_kp = (self.cos_kb, self.sin_kb) if self.cfg.rope_b3 else None
        order = tuple(self.cfg.k_order) or ("a",) * len(self.a3) + ("b",) * len(self.b3)
        stacks = {"a": iter(self.a3), "b": iter(self.b3)}

        k_a3 = k
        for stage in order:
            blk = next(stacks[stage])
            if stage == "a":
                k = blk(k, seq, keep_ctx=keep, rope_ctx=rope_ctx, rope_k=rope_ka, rope_self=rope_kk)
                k_a3 = k
            else:
                k = blk(k, x, rope_ctx=rope_px, rope_k=rope_kp, rope_self=rope_kk)
        rope_q = (self.cos_b4, self.sin_b4) if self.cfg.rope_b4_q else None
        rope_kb = (self.cos_kb, self.sin_kb) if self.cfg.rope_k_b4_k else None

        k_b3, kb = k, k
        rope_b = self._rope_pixel_self(rope_b)
        for blk in self.b4:
            x, kb = blk(x, kb, rope=rope_b, rope_q=rope_q, rope_k=rope_kb, rope_px=rope_px, rope_kp=rope_kp)

        return self._decode(x), k_a3, k_b3

    def _rope_pixel_self(self, rope_b):
        if self.cfg.rope_px_self == "b4":
            return (self.cos_b4, self.sin_b4)
        return rope_b

    def _decode_no_bottleneck(self, seq, keep, rope_a, x, rope_b, grid=None):
        rope_px = (self.cos_b4, self.sin_b4) if self.cfg.rope_b3 else None
        for blk in self.b3:
            seq = blk(seq, x, rope_ctx=rope_px, rope_k=rope_a, rope_self=rope_a, keep_self=keep)
        rope_q = (self.cos_b4, self.sin_b4) if self.cfg.rope_b4_q else None
        rope_b = self._rope_pixel_self(rope_b)
        two_way = self.cfg.decoder == "two_way"
        x0, s0, previous = x, seq, []
        for round_ in range(self.cfg.b4_rounds):
            if round_:
                previous.append(self._decode(x))
                x = x + x0
                if two_way:
                    seq = seq + s0
            for i, blk in enumerate(self.b4):
                if grid is not None and i == self.cfg.grid_cross_before:
                    tok_g, keep_g = grid
                    rope_g = (self.cos_g[: tok_g.shape[1]], self.sin_g[: tok_g.shape[1]])
                    seq = self.grid_cross(
                        seq,
                        tok_g,
                        keep_ctx=keep_g,
                        rope_ctx=rope_g,
                        rope_k=rope_a,
                        rope_self=rope_a,
                        keep_self=keep,
                    )
                if two_way:
                    x, seq = blk(
                        x, seq, keep, rope_a=rope_a, rope_b=rope_b, rope_px_k=rope_px, rope_px_q=rope_q
                    )
                else:
                    x, _ = blk(x, seq, rope=rope_b, rope_q=rope_q, rope_k=rope_a, keep_k=keep)
        self.previous_logits = previous
        return self._decode(x), None, None

    def _forward_depth(self, seq, keep, rope_a, test_in_u8, dt):
        fs = stream_new(seq)
        for blk in self.a2:
            fs = blk(fs, rope=rope_a, keep=keep)
        cond = self.cond_test(self.reader_cond(fs), keep) if self.cond_test is not None else None
        fx = stream_new(self.cnn_test(one_hot_grids(test_in_u8, test_in_u8.shape[0], dt), cond))
        rope_b = (self.cos_b, self.sin_b)
        for blk in self.b2:
            fx = blk(fx, rope=rope_b)
        rope_px = (self.cos_b4, self.sin_b4) if self.cfg.rope_b3 else None
        for blk in self.b3:
            fs = blk(fs, fx, rope_ctx=rope_px, rope_k=rope_a, rope_self=rope_a, keep_self=keep)
        rope_q = (self.cos_b4, self.sin_b4) if self.cfg.rope_b4_q else None
        rope_b = self._rope_pixel_self(rope_b)
        if self.cfg.decoder == "two_way":
            for blk in self.b4:
                fx, fs = blk(fx, fs, keep, rope_a=rope_a, rope_b=rope_b, rope_px_k=rope_px, rope_px_q=rope_q)
        else:
            for blk in self.b4:
                fx, _ = blk(fx, fs, rope=rope_b, rope_q=rope_q, rope_k=rope_a, keep_k=keep)
        self.previous_logits = []
        return self._decode(self.reader_head(fx)), None, None


_PX_ROW = torch.arange(GRID_SIZE * GRID_SIZE) // GRID_SIZE
_PX_COL = torch.arange(GRID_SIZE * GRID_SIZE) % GRID_SIZE

FRAME = (_PX_ROW == 0) | (_PX_ROW == GRID_SIZE - 1) | (_PX_COL == 0) | (_PX_COL == GRID_SIZE - 1)


def without_frame(ce, logits, target_u8, input_u8):
    B = ce.shape[0]
    d = ~FRAME.to(ce.device)
    return (
        ce[:, d],
        logits.reshape(B, -1, N_COLORS)[:, d],
        target_u8.reshape(B, -1)[:, d],
        None if input_u8 is None else input_u8.reshape(B, -1)[:, d],
    )


def pixel_classes(target_u8: torch.Tensor, input_u8: torch.Tensor) -> torch.Tensor:
    B = target_u8.shape[0]
    tgt = target_u8.reshape(B, -1)
    inp = input_u8.reshape(B, -1)
    return 2 * (tgt == inp).long() + (tgt == EMPTY_VALUE).long()


def beta_at(step: int, total_steps: int, start: float, end: float) -> float:
    t = min(max(step / max(1, total_steps), 0.0), 1.0)
    return start + (end - start) * t


def class_weights(cfg: "Config", beta: float, device) -> torch.Tensor:
    return torch.tensor(cfg.loss_class_weights, dtype=torch.float32, device=device) ** beta


def zone_weights(cfg: "Config", device) -> dict | None:
    if not cfg.color_balanced_loss:
        return None
    t = lambda v: torch.tensor(v, dtype=torch.float32, device=device)
    return {
        "zone": t(cfg.loss_zone_weights),
        "zone2": t(cfg.loss_zone_weights_gen),
        "alpha": float(cfg.loss_alpha),
        "empty": float(cfg.empty_weight),
        "task_mass": bool(cfg.loss_task_mass),
    }


def color_balanced_ce(
    ce: torch.Tensor,
    target_u8: torch.Tensor,
    input_u8: torch.Tensor,
    zone_cfg: torch.Tensor,
    group_u8: torch.Tensor | None = None,
    empty_weight_: float = 1.0,
    unit_mass: bool = False,
) -> torch.Tensor:
    """Per-task loss balanced across colors, each weighted by its zone."""
    B = ce.shape[0]
    tgt = target_u8.reshape(B, -1).long()
    inp = input_u8.reshape(B, -1).long()
    grp = tgt if group_u8 is None else group_u8.reshape(B, -1).long()
    n = F.one_hot(grp, N_COLORS).sum(1).float()
    n_colors = (n > 0).sum(1, keepdim=True).float()
    n_c = n.gather(1, grp)
    w = torch.where(tgt != inp, zone_cfg[0], zone_cfg[1])

    w = w * torch.where(grp == EMPTY_VALUE, empty_weight_, 1.0)
    if unit_mass:
        om = w / (n_c * n_colors)
        return (om * ce).sum(1) / om.sum(1).clamp(min=1e-12)
    return (w * ce / (n_c * n_colors)).sum(1)


def color_balanced_loss(
    ce: torch.Tensor,
    logits: torch.Tensor,
    target_u8: torch.Tensor,
    input_u8: torch.Tensor,
    pz: dict,
    parts: dict | None = None,
) -> torch.Tensor:
    """Combines the target-color term and the predicted-color term with weight alpha."""
    B = ce.shape[0]
    gen = logits.detach().reshape(B, -1, N_COLORS).argmax(-1)
    task_mass = pz.get("task_mass", False)
    l_gt = color_balanced_ce(
        ce, target_u8, input_u8, pz["zone"], empty_weight_=pz["empty"], unit_mass=task_mass
    )
    l_gen = color_balanced_ce(
        ce, target_u8, input_u8, pz["zone2"], gen, empty_weight_=pz["empty"], unit_mass=task_mass
    )
    if parts is not None:
        parts["gt"], parts["gen"] = l_gt.detach(), l_gen.detach()
    return pz["alpha"] * l_gt + (1.0 - pz["alpha"]) * l_gen


def compute_loss(
    logits: torch.Tensor,
    target_u8: torch.Tensor,
    input_u8: torch.Tensor | None = None,
    weights: torch.Tensor | None = None,
    zone_cfg: dict | None = None,
    no_frame: bool = False,
    parts: dict | None = None,
) -> tuple:
    """Returns the training loss and the unweighted losses used for logging."""
    B = logits.shape[0]
    tgt = target_u8.reshape(B, -1).long()
    ce = F.cross_entropy(logits.reshape(-1, N_COLORS).float(), tgt.reshape(-1), reduction="none").view(B, -1)
    content = (tgt != EMPTY_VALUE).float()
    shape_ = 1.0 - content
    uniform = ce.mean()

    cw, lw, tw, iw = (
        (ce, logits, target_u8, input_u8) if not no_frame else without_frame(ce, logits, target_u8, input_u8)
    )
    if iw is not None and zone_cfg is not None:
        by_item = {} if parts is not None else None
        loss = color_balanced_loss(cw, lw, tw, iw, zone_cfg, by_item).mean()
        if parts is not None:
            parts.update({k: v.mean() for k, v in by_item.items()})
    elif weights is None or iw is None:
        loss = cw.mean()
    else:
        loss = (weights[pixel_classes(tw, iw)] * cw).mean()
    l_content = (ce * content).sum() / content.sum().clamp(min=1)
    l_shape = (ce * shape_).sum() / shape_.sum().clamp(min=1)
    return loss, l_content.detach(), l_shape.detach(), uniform.detach()


@torch.no_grad()
def batch_rect(plane: torch.Tensor) -> torch.Tensor:
    m = plane.view(-1, GRID_SIZE, GRID_SIZE) != EMPTY_VALUE
    idx = torch.arange(GRID_SIZE, device=plane.device)
    out = []
    for axis_ in (2, 1):
        has_any = m.any(dim=axis_)

        first = torch.where(has_any, idx, GRID_SIZE).amin(dim=1)
        last_ = torch.where(has_any, idx, -1).amax(dim=1)
        out.append(torch.where(has_any.any(dim=1), last_ - first + 1, torch.zeros_like(first)))
    return torch.stack(out, dim=1)


@torch.no_grad()
def pixel_counts(logits: torch.Tensor, target_u8: torch.Tensor, input_u8: torch.Tensor | None = None) -> dict:
    """Per-example counts behind the metrics of the paper."""
    B = logits.shape[0]
    tgt = target_u8.reshape(B, -1).long()
    pred = logits.argmax(-1)
    ok = pred == tgt
    content = tgt != EMPTY_VALUE
    gen_empty = pred == EMPTY_VALUE
    gt_empty = ~content
    extra = {}
    if input_u8 is not None:
        inp = input_u8.reshape(B, -1).long()
        r_pred, r_tgt, r_inp = batch_rect(pred), batch_rect(tgt), batch_rect(inp)
        cls = pixel_classes(tgt, inp)
        cc, cv = cls == 0, cls == 1
        junk = cls >= 2

        count_ = F.one_hot(inp, N_COLORS)[..., :EMPTY_VALUE].sum(1)
        is_background = tgt == count_.argmax(1, keepdim=True)
        cc_bg, cc_nonbg = cc & is_background, cc & ~is_background
        extra = {
            "ok_ccf": (ok & cc_bg).float().sum(1),
            "n_ccf": cc_bg.float().sum(1),
            "ok_ccnf": (ok & cc_nonbg).float().sum(1),
            "n_ccnf": cc_nonbg.float().sum(1),
            "ok_cc": (ok & cc).float().sum(1),
            "n_cc": cc.float().sum(1),
            "ok_cv": (ok & cv).float().sum(1),
            "n_cv": cv.float().sum(1),
            "ok_junk": (ok & junk).float().sum(1),
            "n_junk": junk.float().sum(1),
            "changed_cells": (pred != inp).float().sum(1),
            "target_changed_cells": (tgt != inp).float().sum(1),
            "shape_match": (r_pred == r_tgt).all(dim=1).float(),
            "diff_shape": (~(r_inp == r_tgt).all(dim=1)).float(),
        }
    return {
        **extra,
        "exact": ok.all(dim=1).float(),
        "shape_error": (~gen_empty & gt_empty).float().sum(1),
        "fill_error": (gen_empty & content).float().sum(1),
        "ok_total": ok.float().sum(1),
        "n_total": torch.full_like(ok.float().sum(1), ok.shape[1]),
        "ok_cont": (ok & content).float().sum(1),
        "n_cont": content.float().sum(1),
        "ok_shape": (ok & ~content).float().sum(1),
        "n_shape": (~content).float().sum(1),
    }


def _rect(g) -> tuple:
    m = np.asarray(g) != EMPTY_VALUE
    row_idx, column = np.where(m.any(axis=1))[0], np.where(m.any(axis=0))[0]
    if len(row_idx) == 0:
        return (0, 0)
    return (int(row_idx[-1] - row_idx[0] + 1), int(column[-1] - column[0] + 1))


def _same_shape(inp, gt) -> bool:
    return _rect(inp) == _rect(gt)


def _orthogonalize(G: torch.Tensor, steps: int = 5, eps: float = 1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    tall = G.size(-2) > G.size(-1)
    X = G.mT if tall else G
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


class Muon(torch.optim.Optimizer):
    """Muon for matrices and flattened convolution filters, with updates scaled to the RMS of AdamW."""

    def __init__(
        self, params, lr: float, weight_decay: float, momentum: float = 0.95, ns_dtype=torch.bfloat16
    ):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay, momentum=momentum))
        self.ns_dtype = ns_dtype

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, wd, mu = g["lr"], g["weight_decay"], g["momentum"]
            ps = [p for p in g["params"] if p.grad is not None]
            if not ps:
                continue
            grads = [p.grad for p in ps]
            bufs = []
            for p in ps:
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p)
                bufs.append(st["buf"])
            torch._foreach_mul_(bufs, mu)
            torch._foreach_add_(bufs, grads)
            upd = torch._foreach_add(grads, bufs, alpha=mu)
            torch._foreach_mul_(ps, 1 - lr * wd)
            by_shape = {}
            for p, u in zip(ps, upd):
                m = u.reshape(u.shape[0], -1)
                by_shape.setdefault(tuple(m.shape), []).append((p, m))
            for (A, B), items in by_shape.items():
                O = _orthogonalize(torch.stack([m for _, m in items]).to(self.ns_dtype))
                torch._foreach_add_(
                    [p for p, _ in items],
                    [o.reshape(p.shape).to(p.dtype) for (p, _), o in zip(items, O)],
                    alpha=-lr * 0.2 * math.sqrt(max(A, B)),
                )


class HybridOptimizer:
    """Muon for hidden matrices and AdamW for the remaining parameters."""

    def __init__(self, muon: Muon, adamw: torch.optim.AdamW):
        self.parts = {"muon": muon, "adamw": adamw}

    @property
    def param_groups(self):
        return [g for o in self.parts.values() for g in o.param_groups]

    def zero_grad(self, set_to_none: bool = True):
        for o in self.parts.values():
            o.zero_grad(set_to_none=set_to_none)

    def step(self):
        for o in self.parts.values():
            o.step()

    def state_dict(self):
        return {k: o.state_dict() for k, o in self.parts.items()}

    def load_state_dict(self, sd):
        for k, o in self.parts.items():
            o.load_state_dict(sd[k])


def muon_param_split(model: nn.Module) -> tuple:
    """Splits parameters into those trained with Muon and those trained with AdamW."""
    excluded = {id(m.entry.weight) for m in model.modules() if isinstance(m, (GridCNN, DemoStem))}
    excluded |= {id(p) for p in model.head.parameters()}
    if model.cfg.attn_gate_adamw:
        excluded |= {id(m.g.weight) for m in model.modules() if isinstance(m, Attention) and m.gate}
    muon = [
        m.weight
        for m in model.modules()
        if (isinstance(m, nn.Linear) or (isinstance(m, nn.Conv2d) and m.groups == 1))
        and id(m.weight) not in excluded
    ]
    ids = {id(p) for p in muon}
    return muon, [p for p in model.parameters() if id(p) not in ids]


def adamw_groups(params, cfg: Config) -> list:
    params = list(params)
    if cfg.decay_1d:
        return [{"params": params, "weight_decay": cfg.weight_decay}]
    return [
        {"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
        {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, cfg: Config):
    """Builds the optimizer selected by the configuration."""
    if cfg.optimizer_name == "adamw":
        return torch.optim.AdamW(adamw_groups(model.parameters(), cfg), lr=cfg.lr)
    muon, rest = muon_param_split(model)

    ns = (
        torch.bfloat16
        if cfg.device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
        else torch.float32
    )
    n_m = sum(p.numel() for p in muon)
    n_r = sum(p.numel() for p in rest)
    print(
        f"Optimizer: Muon {len(muon)} matrices ({n_m / 1e6:.2f}M) | "
        f"AdamW {len(rest)} tensors ({n_r / 1e6:.2f}M) | Newton-Schulz in "
        f"{str(ns).split('.')[-1]} | decay on 1D parameters: {'si' if cfg.decay_1d else 'no'}"
    )
    return HybridOptimizer(
        Muon(muon, lr=cfg.lr, weight_decay=cfg.weight_decay, ns_dtype=ns),
        torch.optim.AdamW(adamw_groups(rest, cfg), lr=cfg.lr),
    )


def lr_at(
    step: int, total_steps: int, warmup_steps: int, lr: float, lr_min: float, shape_: str = "cosine"
) -> float:
    """Learning rate at a given step: warmup followed by the configured decay."""
    if warmup_steps > 0 and step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    t = min(max(t, 0.0), 1.0)
    if shape_ == "linear":
        return lr * (1.0 - t)
    return lr_min + 0.5 * (lr - lr_min) * (1.0 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(
    model: CAIL, loader: DataLoader, cfg: Config, amp_dtype, weights: torch.Tensor | None = None
) -> dict:
    """Evaluates the model and returns the metrics and the indices of the solved items."""
    device = cfg.device
    model.eval()
    zone_cfg = zone_weights(cfg, device)

    FIELDS = (
        "ce",
        "ce_weighted",
        "ce_cont",
        "ce_shape",
        "exact",
        "ok_total",
        "n_total",
        "ok_cont",
        "n_cont",
        "ok_shape",
        "n_shape",
        "ok_cc",
        "n_cc",
        "ok_cv",
        "n_cv",
        "ok_junk",
        "n_junk",
        "ok_ccf",
        "n_ccf",
        "ok_ccnf",
        "n_ccnf",
        "shape_error",
        "fill_error",
        "changed_cells",
        "target_changed_cells",
        "shape_match",
        "diff_shape",
    )
    FIELDS += tuple(f"ok_cc_v{j}" for j in range(1, cfg.b4_rounds))
    if zone_cfg is not None:
        FIELDS += ("ce_gt", "ce_gen")
    acc = {k: [] for k in FIELDS}
    groups_list, tasks_list = [], []

    ds = getattr(loader, "dataset", None)
    tasks = getattr(ds, "tasks", None)

    for demo_in, demo_out, test_in, tgt, n_demos, group, item in loader:
        demo_in = demo_in.to(device, non_blocking=True)
        demo_out = demo_out.to(device, non_blocking=True)
        test_in = test_in.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=cfg.use_amp and device == "cuda", dtype=amp_dtype):
            logits, k_a3, k_upd = model(demo_in, demo_out, test_in, n_demos.tolist())

        B = logits.shape[0]
        t = tgt.reshape(B, -1).long()
        ce = F.cross_entropy(logits.reshape(-1, N_COLORS).float(), t.reshape(-1), reduction="none").view(
            B, -1
        )
        content = (t != EMPTY_VALUE).float()

        cw, lw, tw, iw = (
            (ce, logits, tgt, test_in) if not cfg.free_frame else without_frame(ce, logits, tgt, test_in)
        )
        parts = {}
        if zone_cfg is not None:
            ce_weighted = color_balanced_loss(cw, lw, tw, iw, zone_cfg, parts)
        else:
            ce_weighted = (cw if weights is None else weights[pixel_classes(tw, iw)] * cw).mean(1)
        a = pixel_counts(logits, tgt, test_in)

        row_ = {
            "ce": ce.mean(1),
            "ce_weighted": ce_weighted,
            "ce_cont": (ce * content).sum(1) / content.sum(1).clamp(min=1),
            "ce_shape": (ce * (1 - content)).sum(1) / (1 - content).sum(1).clamp(min=1),
            "exact": a["exact"],
            "ok_total": a["ok_total"],
            "n_total": a["n_total"],
            "ok_cont": a["ok_cont"],
            "n_cont": a["n_cont"],
            "ok_shape": a["ok_shape"],
            "n_shape": a["n_shape"],
            "ok_cc": a["ok_cc"],
            "n_cc": a["n_cc"],
            "ok_cv": a["ok_cv"],
            "n_cv": a["n_cv"],
            "ok_junk": a["ok_junk"],
            "n_junk": a["n_junk"],
            "ok_ccf": a["ok_ccf"],
            "n_ccf": a["n_ccf"],
            "ok_ccnf": a["ok_ccnf"],
            "n_ccnf": a["n_ccnf"],
            "shape_error": a["shape_error"],
            "fill_error": a["fill_error"],
            "changed_cells": a["changed_cells"],
            "target_changed_cells": a["target_changed_cells"],
            "shape_match": a["shape_match"],
            "diff_shape": a["diff_shape"],
        }
        for k_, v_ in parts.items():
            row_[f"ce_{k_}"] = v_
        for j, lp in enumerate(model.previous_logits, start=1):
            row_[f"ok_cc_v{j}"] = pixel_counts(lp, tgt, test_in)["ok_cc"]
        for k, v in row_.items():
            acc[k].append(v.float().cpu())
        groups_list.append(group)
        tasks_list.append(item)

    d = {k: torch.cat(v) for k, v in acc.items()}
    group_t = torch.cat(groups_list)
    item_ids = torch.cat(tasks_list)

    diff_shape_mask = d["diff_shape"].bool()

    def summarize(m: torch.Tensor, suffix: str = "") -> dict:
        n = max(int(m.sum()), 1)
        s = lambda k: float(d[k][m].sum())
        out = {
            f"loss{suffix}": s("ce_weighted") / n,
            f"loss_uniform{suffix}": s("ce") / n,
            f"loss_content{suffix}": s("ce_cont") / n,
            f"loss_shape{suffix}": s("ce_shape") / n,
            f"exact_match{suffix}": s("exact") / n,
            f"exact_match_n{suffix}": s("exact"),
            f"PA{suffix}": s("ok_total") / max(s("n_total"), 1),
            f"CA{suffix}": s("ok_cont") / max(s("n_cont"), 1),
            f"shape_accuracy{suffix}": s("ok_shape") / max(s("n_shape"), 1),
            f"changed_accuracy{suffix}": (s("ok_cc") + s("ok_cv")) / max(s("n_cc") + s("n_cv"), 1),
            f"CC{suffix}": s("ok_cc") / max(s("n_cc"), 1),
            f"changed_to_empty_accuracy{suffix}": s("ok_cv") / max(s("n_cv"), 1),
            f"UA{suffix}": s("ok_junk") / max(s("n_junk"), 1),
            f"CC_background{suffix}": s("ok_ccf") / max(s("n_ccf"), 1),
            f"CC_nonbackground{suffix}": s("ok_ccnf") / max(s("n_ccnf"), 1),
            f"shape_error{suffix}": s("shape_error") / n,
            f"fill_error{suffix}": s("fill_error") / n,
            f"frac_diff_shape{suffix}": s("diff_shape") / n,
        }

        for k_ in ("gt", "gen"):
            if f"ce_{k_}" in d:
                out[f"loss_color_{k_}{suffix}"] = s(f"ce_{k_}") / n

        for label, mm in (
            ("", m),
            ("_same_shape", m & ~diff_shape_mask),
            ("_diff_shape", m & diff_shape_mask),
        ):
            k = max(int(mm.sum()), 1)
            out[f"changed_cells{label}{suffix}"] = float(d["changed_cells"][mm].sum()) / k
            out[f"target_changed_cells{label}{suffix}"] = float(d["target_changed_cells"][mm].sum()) / k
            out[f"shape_match{label}{suffix}"] = float(d["shape_match"][mm].sum()) / k
        for key in d:
            if key.startswith("ok_cc_v"):
                j = key.removeprefix("ok_cc_v")
                out[f"CC_v{j}{suffix}"] = s(key) / max(s("n_cc"), 1)
        return out

    m_arc = group_t == SUBSET_ARC
    m_total = torch.isin(group_t, torch.tensor(SUBSETS_IN_TOTAL))
    out = summarize(m_total)
    for g, suffix in SUBSET_SUFFIX.items():
        out.update(summarize(group_t == g, suffix))
    out["n_eval"] = float(m_total.sum())

    if tasks is not None:
        by_task = {}
        for e, i in zip(d["exact"][m_arc].tolist(), item_ids[m_arc].tolist()):
            t = tasks[int(i)]
            by_task[t] = min(by_task.get(t, 1.0), e)
        out["exact_match_n_arc_task"] = float(sum(by_task.values()))
        out["exact_match_arc_task"] = out["exact_match_n_arc_task"] / max(len(by_task), 1)
        out["n_eval_arc_tasks"] = float(len(by_task))

    solved = [int(i) for i, e in zip(item_ids.tolist(), d["exact"].tolist()) if e >= 1.0]
    model.train()
    return out, solved


_PIXEL_RANDOM = 1.0 / N_COLORS


def verify_pipeline(cfg: Config, dataset: Dataset, device: str) -> bool:
    """Checks that the optimized one-hot pipeline matches a naive reference exactly."""
    items = [dataset[i] for i in range(min(4, len(dataset)))]

    def naive(u8: torch.Tensor) -> torch.Tensor:
        g = u8.numpy()
        out = np.zeros((g.shape[0], N_COLORS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        for i in range(g.shape[0]):
            for c in range(N_COLORS):
                out[i, c] = (g[i] == c).astype(np.float32)
        return torch.from_numpy(out)

    demo_in, demo_out, test_in, tgt, n_demos, _, _ = collate_fn(items)
    ok = True
    for name_, u8 in (("demos in", demo_in), ("demos out", demo_out), ("test in", test_in)):
        ref = naive(u8).to(device)
        n_real = ref.shape[0]
        n_pad = _round_up(n_real, CNN_BATCH_MULTIPLE)
        if n_pad > n_real:
            ref = torch.cat([ref, ref.new_zeros((n_pad - n_real, *ref.shape[1:]))], 0)
        got = one_hot_grids(u8.to(device), n_pad)
        if not torch.equal(ref, got):
            ok = False
            print(f"  {name_}: {(ref != got).sum().item()} positions differ out of {ref.numel()}")
    print(f"verify_pipeline    : {'OK' if ok else 'FAILED'} (exact equality)")
    return ok


FIG_BG, FIG_FG, FIG_MUTE, GRID_LN = "#242428", "#e8e6e3", "#9b98a3", "#6e6b76"


def ckpt_criterion(ev: dict) -> tuple:
    """Selects the best checkpoint by CC over the evaluation set."""
    return (float(ev.get("CC", -1.0)),)


def save_checkpoint(path, epoch, step, model, optimizer, cfg, extra):
    """Saves the model, the optimizer and the configuration."""
    torch.save(
        {
            "epoch": epoch,
            "global_step": step,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "cfg": asdict(cfg),
            **extra,
        },
        path,
    )


class RunLogger:
    """Writes `meta.json`, `train.jsonl` and `eval.jsonl` into the run directory."""

    def __init__(self, run_dir: str):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def meta(self, **fields):
        (self.dir / "meta.json").write_text(json.dumps(fields, indent=2, default=str))

    def log(self, stream: str, **row):
        with open(self.dir / f"{stream}.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    def rows(self, stream: str) -> list:
        path = self.dir / f"{stream}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _count_summary(counts: dict, losses: dict, n_batches: int) -> dict:
    d = lambda a, b: counts[a] / max(counts[b], 1)
    return {
        "loss": losses["loss"] / max(n_batches, 1),
        "exact_match": counts["exact"] / max(counts["items"], 1),
        "PA": d("ok_total", "n_total"),
        "CA": d("ok_cont", "n_cont"),
        "CC": d("ok_cc", "n_cc"),
        "UA": d("ok_junk", "n_junk"),
        "CC_background": d("ok_ccf", "n_ccf"),
        "CC_nonbackground": d("ok_ccnf", "n_ccnf"),
    }


COUNT_KEYS = (
    "ok_total",
    "n_total",
    "ok_cont",
    "n_cont",
    "ok_cc",
    "n_cc",
    "ok_junk",
    "n_junk",
    "ok_ccf",
    "n_ccf",
    "ok_ccnf",
    "n_ccnf",
)


def train(cfg: Config) -> dict:
    """Trains one run and writes its JSON logs and checkpoints to `cfg.run_dir`."""
    validate(cfg)
    device = cfg.device
    set_seed(cfg.seed)
    logger = RunLogger(cfg.run_dir)

    arc_tasks = load_arc(cfg.arc_train_dir, cfg.min_pairs)
    split = (
        split_nvarc(
            cfg.nvarc_dir,
            cfg.arc_eval_dir,
            cfg.nvarc_heldout_n,
            cfg.nvarc_split_seed,
            cfg.nvarc_families_heldout,
            cfg.nvarc_families_seed,
            cfg.nvarc_tasks_per_family,
        )
        if cfg.use_nvarc
        else NvarcSplit(train=[], files=[], families=[], selected_families=[], n_families_total=0)
    )
    arc_train = arc_without_families(arc_tasks, split.selected_families)
    full_eval = select_nvarc_full(cfg.nvarc_full_dir, cfg.nvarc_full_eval_n, cfg.nvarc_full_seed)
    nvarc_train = list(split.train)
    if cfg.train_with_nvarc_full:
        excluded = set(full_eval)
        nvarc_train += [
            str(f) for f in sorted(Path(cfg.nvarc_full_dir).glob("*/*.json")) if str(f) not in excluded
        ]

    dataset = TrainDataset(cfg, arc_tasks=arc_train, nvarc_files=nvarc_train)
    generator = torch.Generator()
    generator.manual_seed(cfg.seed)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory and device == "cuda",
        persistent_workers=cfg.num_workers > 0,
        drop_last=True,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    eval_set = EvalSet(cfg, split=split, full=full_eval)
    eval_loader = DataLoader(
        eval_set,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=cfg.pin_memory and device == "cuda",
        drop_last=False,
    )
    if cfg.verify_pipeline and not verify_pipeline(cfg, dataset, device):
        raise RuntimeError("Internal consistency check failed.")

    model = CAIL(cfg).to(device)
    if cfg.compile_model and device == "cuda":
        model.compile_model()
    amp_dtype = (
        torch.bfloat16
        if device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
        else torch.float16
    )
    use_amp = cfg.use_amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    optimizer = build_optimizer(model, cfg)

    steps_per_epoch = max(1, len(loader))
    total_steps = (cfg.sched_epochs or cfg.epochs) * steps_per_epoch
    warmup_steps = int(round(cfg.warmup_epochs * steps_per_epoch))
    start_epoch, best, global_step = 0, (-1.0,), 0
    solved_ever: set = set()

    if os.path.exists(cfg.backup_path):
        ckpt = torch.load(cfg.backup_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        best = ckpt.get("best", best)
        solved_ever = set(ckpt.get("solved_ever", []))
        print(f"Resumed from epoch {start_epoch}")

    n_params = sum(p.numel() for p in model.parameters())
    logger.meta(
        cfg=asdict(cfg),
        n_params=n_params,
        n_train_tasks=len(dataset),
        n_train_arc=len(dataset.arc),
        n_train_nvarc=len(dataset.nvarc),
        n_eval_items=len(eval_set),
        steps_per_epoch=steps_per_epoch,
        total_sched_steps=total_steps,
        warmup_steps=warmup_steps,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    )
    print(
        f"Parameters: {n_params:,} | training tasks: {len(dataset)} | eval items: {len(eval_set)} | seed: {cfg.seed}"
    )

    last_eval = {}
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        counts = {k: 0.0 for k in COUNT_KEYS + ("exact", "items")}
        losses = {"loss": 0.0}
        n_batches = 0
        lr = lr_at(global_step, total_steps, warmup_steps, cfg.lr, cfg.lr_floor, cfg.lr_shape)
        beta = beta_at(global_step, total_steps, cfg.loss_beta_start, cfg.loss_beta_end)
        weights = class_weights(cfg, beta, device)
        zone_cfg = zone_weights(cfg, device)
        t0 = time.time()
        for demo_in, demo_out, test_in, tgt, n_demos, _, _ in loader:
            if global_step % cfg.sched_every == 0:
                lr = lr_at(global_step, total_steps, warmup_steps, cfg.lr, cfg.lr_floor, cfg.lr_shape)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                beta = beta_at(global_step, total_steps, cfg.loss_beta_start, cfg.loss_beta_end)
                weights = class_weights(cfg, beta, device)
            demo_in, demo_out = demo_in.to(device, non_blocking=True), demo_out.to(device, non_blocking=True)
            test_in, tgt = test_in.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits, _, _ = model(demo_in, demo_out, test_in, n_demos.tolist())
                loss, _, _, _ = compute_loss(logits, tgt, test_in, weights, zone_cfg, cfg.free_frame, {})
                if model.previous_logits:
                    extra = [
                        compute_loss(p, tgt, test_in, weights, zone_cfg, cfg.free_frame)[0]
                        for p in model.previous_logits
                    ]
                    loss = (loss + sum(extra)) / (len(extra) + 1)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            a = pixel_counts(logits, tgt, test_in)
            for k in COUNT_KEYS + ("exact",):
                counts[k] += float(a[k].sum())
            counts["items"] += logits.shape[0]
            losses["loss"] += loss.item()
            n_batches += 1
            global_step += 1
        elapsed = time.time() - t0
        logger.log(
            "train",
            epoch=epoch + 1,
            global_step=global_step,
            lr=lr,
            beta=beta,
            epoch_time_s=elapsed,
            **_count_summary(counts, losses, n_batches),
        )

        if (epoch + 1) % cfg.eval_every == 0 or epoch + 1 == cfg.epochs:
            t_eval = time.time()
            ev, solved = evaluate(model, eval_loader, cfg, amp_dtype, weights)
            solved_ever |= set(solved)
            ev.update(
                epoch=epoch + 1,
                beta=beta,
                eval_time_s=time.time() - t_eval,
                solved_items=solved,
                exact_ever_solved=len(solved_ever),
            )
            logger.log("eval", **ev)
            last_eval = ev
            score = ckpt_criterion(ev)
            improved = score > best
            best = max(best, score)
            if cfg.save_ckpt:
                extra = {"best": best, "solved_ever": sorted(solved_ever)}
                save_checkpoint(cfg.backup_path, epoch, global_step, model, optimizer, cfg, extra)
                if improved:
                    save_checkpoint(cfg.save_path, epoch, global_step, model, optimizer, cfg, extra)
            print(
                f"[epoch {epoch + 1}/{cfg.epochs}] train loss {losses['loss'] / max(n_batches, 1):.3f} | eval loss {ev['loss']:.3f} | "
                f"exact {int(ev['exact_match_n'])} | CC {ev['CC']:.4f} | CA {ev['CA']:.4f} | PA {ev['PA']:.4f} | {ev['eval_time_s']:.0f}s",
                flush=True,
            )
    return last_eval


def load_config(path: str, data_dir: str, out_dir: str, run_name: str = "") -> Config:
    """Builds a Config from a JSON file, rooted at `data_dir` and `out_dir`."""
    values = json.loads(Path(path).read_text())
    root = Path(data_dir)
    values.update(
        arc_train_dir=str(root / "arc-agi-2/data/training"),
        arc_eval_dir=str(root / "arc-agi-2/data/evaluation"),
        nvarc_dir=str(root / "nvarc_training"),
        nvarc_full_dir=str(root / "nvarc_full"),
        out_dir=out_dir,
    )
    if run_name:
        values["run_name"] = run_name
    return Config(**values)


def main():
    parser = argparse.ArgumentParser(description="Train CAIL from a JSON configuration.")
    parser.add_argument("--config", required=True, help="Path to a configuration in configs/.")
    parser.add_argument(
        "--data-dir", default="data", help="Directory with arc-agi-2, nvarc_training and nvarc_full."
    )
    parser.add_argument("--out-dir", default="runs")
    parser.add_argument("--run-name", default="")
    args = parser.parse_args()
    train(load_config(args.config, args.data_dir, args.out_dir, args.run_name))


if __name__ == "__main__":
    main()
