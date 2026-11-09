"""Test-time augmentation by voting over symmetries and color permutations.

Each view applies one of the 8 symmetries of the square and a permutation of the colors 1-9 to a whole
task (black and empty stay fixed). The prediction of each view is mapped back and counts as one vote for
that grid. The two most voted grids are the two allowed attempts; ties are broken by summed confidence.

Usage: python vote.py --run-dir runs/final_model --data-dir data --n-colors 125
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from cail import CAIL, Config, EMPTY_VALUE, EvalSet, GRID_SIZE, N_COLORS, collate_fn

SYMMETRIES = {
    "id": (lambda a: a, lambda a: a),
    "rot90": (lambda a: np.rot90(a, 1), lambda a: np.rot90(a, 3)),
    "rot180": (lambda a: np.rot90(a, 2), lambda a: np.rot90(a, 2)),
    "rot270": (lambda a: np.rot90(a, 3), lambda a: np.rot90(a, 1)),
    "flip_h": (lambda a: a[:, ::-1], lambda a: a[:, ::-1]),
    "flip_v": (lambda a: a[::-1], lambda a: a[::-1]),
    "transpose": (lambda a: a.T, lambda a: a.T),
    "anti_transpose": (lambda a: np.rot90(a, 2).T, lambda a: np.rot90(a.T, 2)),
}
SUBSETS = ("ARC", "NVARC", "FAMILIES", "FULL")


def crop(g: np.ndarray) -> np.ndarray:
    """Returns the non-empty rectangle of a padded grid."""
    mask = g != EMPTY_VALUE
    if not mask.any():
        return g[:0, :0]
    rows, cols = np.where(mask.any(1))[0], np.where(mask.any(0))[0]
    return g[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def repad(a: np.ndarray) -> np.ndarray:
    """Places a grid back on the 32x32 canvas at (1, 1), truncating to 31 cells if needed."""
    out = np.full((GRID_SIZE, GRID_SIZE), EMPTY_VALUE, np.uint8)
    h, w = min(a.shape[0], GRID_SIZE - 1), min(a.shape[1], GRID_SIZE - 1)
    out[1 : h + 1, 1 : w + 1] = a[:h, :w]
    return out


def color_tables(n: int, seed: int = 0) -> list:
    """n color lookup tables: the identity first, then distinct permutations of the colors 1-9."""
    rng = np.random.RandomState(seed)
    out = [np.arange(N_COLORS, dtype=np.uint8)]
    while len(out) < n:
        lut = np.arange(N_COLORS, dtype=np.uint8)
        lut[1:10] = 1 + rng.permutation(9)
        if not any((lut == o).all() for o in out):
            out.append(lut)
    return out


class View:
    """One symmetry combined with one color permutation."""

    def __init__(self, symmetry: str, lut: np.ndarray):
        self.symmetry, self.lut = symmetry, lut
        self.inverse_lut = np.argsort(lut).astype(np.uint8)
        self.forward, self.backward = SYMMETRIES[symmetry]

    def __call__(self, g: np.ndarray) -> np.ndarray:
        return repad(np.ascontiguousarray(self.forward(self.lut[crop(g)])))

    def undo(self, p: np.ndarray) -> np.ndarray:
        c = crop(p)
        if c.size == 0:
            return np.full((GRID_SIZE, GRID_SIZE), EMPTY_VALUE, np.uint8)
        return repad(np.ascontiguousarray(self.inverse_lut[self.backward(c)]))


def build_views(n_colors: int, seed: int = 0) -> list:
    return [View(s, lut) for lut in color_tables(n_colors, seed) for s in SYMMETRIES]


@torch.no_grad()
def predict(model, items: list, view: View, batch: int, device: str, dtype) -> tuple:
    """Returns the predicted grids and their mean log-confidence for one view."""
    n = len(items)
    preds = np.empty((n, GRID_SIZE, GRID_SIZE), np.uint8)
    conf = np.empty(n, np.float32)
    for s in range(0, n, batch):
        chunk = []
        for di, do, ti, to, group, idx in items[s : s + batch]:
            chunk.append(
                (
                    torch.from_numpy(np.stack([view(g) for g in di])),
                    torch.from_numpy(np.stack([view(g) for g in do])),
                    torch.from_numpy(view(ti)),
                    torch.from_numpy(view(to)),
                    group,
                    idx,
                )
            )
        demo_in, demo_out, test_in, _, n_demos, _, _ = collate_fn(chunk)
        with torch.autocast("cuda", dtype=dtype, enabled=device == "cuda"):
            logits = model(demo_in.to(device), demo_out.to(device), test_in.to(device), n_demos.tolist())[0]
        best, argmax = logits.float().log_softmax(-1).max(-1)
        preds[s : s + len(chunk)] = argmax.view(-1, GRID_SIZE, GRID_SIZE).cpu().numpy()
        conf[s : s + len(chunk)] = best.mean(-1).cpu().numpy()
    return preds, conf


def vote(model, eval_set, views: list, batch: int, device: str) -> list:
    """Returns one row per test input with single-pass, voted and any-view results."""
    bf16 = device == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    dtype = torch.bfloat16 if bf16 else torch.float16
    items = eval_set.items
    n = len(items)
    votes = [defaultdict(int) for _ in range(n)]
    confidence = [defaultdict(float) for _ in range(n)]
    base = None
    t0 = time.time()
    for k, view in enumerate(views):
        preds, conf = predict(model, items, view, batch, device, dtype)
        if k == 0:
            assert view.symmetry == "id" and (view.lut == np.arange(N_COLORS)).all()
            base = preds.copy()
        for i in range(n):
            key = view.undo(preds[i]).tobytes()
            votes[i][key] += 1
            confidence[i][key] += float(conf[i])
        print(f"view {k + 1}/{len(views)} | {time.time() - t0:.0f}s", flush=True)
    rows = []
    for i, (_, _, ti, to, _, _) in enumerate(items):
        target = to.tobytes()
        order = sorted(votes[i], key=lambda c: (-votes[i][c], -confidence[i][c]))
        info = eval_set.info[i]
        rows.append(
            {
                "subset": info["subset"],
                "task_id": info["id"],
                "pair": info["pair"],
                "task": int(eval_set.tasks[i]),
                "single_pass": bool((base[i] == to).all()),
                "first_attempt": order[0] == target,
                "two_attempts": target in order[:2],
                "any_view": target in votes[i],
            }
        )
    return rows


def summarize(rows: list) -> dict:
    out = {}
    for subset in SUBSETS:
        selected = [r for r in rows if r["subset"] == subset]
        out[subset] = {"items": len(selected)}
        for key in ("single_pass", "first_attempt", "two_attempts", "any_view"):
            out[subset][key] = sum(r[key] for r in selected)
    by_task = defaultdict(list)
    for r in rows:
        if r["subset"] == "ARC":
            by_task[r["task"]].append(r["two_attempts"])
    out["arc_score"] = sum(sum(v) / len(v) for v in by_task.values()) / max(len(by_task), 1)
    return out


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained run with test-time augmentation by voting.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--n-colors", type=int, default=125, help="Color permutations; 125 gives 1,000 views.")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(args.run_dir)
    ckpt = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=False)
    values = dict(ckpt["cfg"])
    root = Path(args.data_dir)
    values.update(
        arc_train_dir=str(root / "arc-agi-2/data/training"),
        arc_eval_dir=str(root / "arc-agi-2/data/evaluation"),
        nvarc_dir=str(root / "nvarc_training"),
        nvarc_full_dir=str(root / "nvarc_full"),
        device=device,
        compile_model=False,
    )
    cfg = Config(**values)
    model = CAIL(cfg)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model = model.to(device).eval()

    eval_set = EvalSet(cfg)
    views = build_views(args.n_colors)
    print(f"{len(views)} views: {len(SYMMETRIES)} symmetries x {args.n_colors} color permutations", flush=True)
    t0 = time.time()
    rows = vote(model, eval_set, views, args.batch, device)
    summary = summarize(rows)
    summary["n_views"] = len(views)
    summary["time_s"] = round(time.time() - t0, 1)
    (run_dir / "vote_summary.json").write_text(json.dumps(summary, indent=2))
    with open(run_dir / "vote_items.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
