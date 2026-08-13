#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PATTERN = re.compile(r"eval_(test|conflict)_corruption(0|25|50|100)_seed(\d+)$")


def interval(values: np.ndarray) -> list[float]:
    return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]


def load_predictions() -> pd.DataFrame:
    frames = []
    for path in sorted((ROOT / "outputs/model_study").glob("eval_*_corruption*_seed*")):
        match = PATTERN.fullmatch(path.name)
        if not match or not (path / "manifest.json").exists():
            continue
        manifest = json.loads((path / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            continue
        frame = pd.read_parquet(path / "predictions.parquet")
        frame["split"] = match.group(1)
        frame["corruption"] = int(match.group(2))
        frame["seed"] = int(match.group(3))
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("no completed sealed model evaluations found")
    data = pd.concat(frames, ignore_index=True)
    expected = {(split, corruption, seed) for split in ("test", "conflict")
                for corruption in (0, 25, 50, 100)
                for seed in (20260808, 20260809, 20260810)}
    observed = set(zip(data.split, data.corruption, data.seed))
    if observed != expected:
        raise RuntimeError(f"incomplete matrix: missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="analysis/model_study")
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()
    output = ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    data = load_predictions()

    per_seed = data.groupby(["split", "corruption", "seed"], as_index=False).agg(
        records=("record_id", "size"),
        exact=("exact", "mean"),
        copied_stale=("copied_stale", "mean"),
        value_token_nll=("value_token_nll", "mean"),
        whole_target_nll=("whole_target_nll", "mean"),
    )
    per_seed.to_csv(output / "per_seed.csv", index=False)
    aggregate = per_seed.groupby(["split", "corruption"], as_index=False).agg(
        seeds=("seed", "size"),
        exact_mean=("exact", "mean"),
        exact_seed_sd=("exact", "std"),
        copied_stale_mean=("copied_stale", "mean"),
        copied_stale_seed_sd=("copied_stale", "std"),
        value_token_nll_mean=("value_token_nll", "mean"),
        value_token_nll_seed_sd=("value_token_nll", "std"),
        whole_target_nll_mean=("whole_target_nll", "mean"),
        whole_target_nll_seed_sd=("whole_target_nll", "std"),
    )
    aggregate.to_csv(output / "aggregate.csv", index=False)

    rng = np.random.default_rng(args.seed)
    contrasts: dict[str, dict] = {}
    for split in ("test", "conflict"):
        subset = data[data.split == split]
        record_ids = np.array(sorted(subset.record_id.unique()))
        corruption_order = [0, 25, 50, 100]
        seed_order = [20260808, 20260809, 20260810]

        def cube(metric: str) -> np.ndarray:
            pivot = subset.pivot(index="record_id", columns=["corruption", "seed"], values=metric)
            pivot = pivot.reindex(index=record_ids)
            ordered = pivot.reindex(
                columns=pd.MultiIndex.from_product(
                    [corruption_order, seed_order], names=["corruption", "seed"]
                )
            )
            if ordered.isna().any().any():
                raise RuntimeError(f"missing paired {metric} result in {split}")
            return ordered.to_numpy(dtype=float).reshape(
                len(record_ids), len(corruption_order), len(seed_order)
            )

        exact_cube = cube("exact")
        nll_cube = cube("value_token_nll")
        stale_cube = cube("copied_stale")
        point = subset.groupby("corruption").agg(
            exact=("exact", "mean"), value_nll=("value_token_nll", "mean"),
            stale=("copied_stale", "mean")
        )
        point_exact_delta = float(point.loc[100, "exact"] - point.loc[0, "exact"])
        point_nll_delta = float(point.loc[100, "value_nll"] - point.loc[0, "value_nll"])
        point_stale_delta = float(point.loc[100, "stale"] - point.loc[0, "stale"])
        exact_draws, nll_draws, stale_draws, exact_slopes, nll_slopes = [], [], [], [], []
        x = np.array([0.0, 0.25, 0.5, 1.0])
        for _ in range(args.bootstrap_replicates):
            sampled_indices = rng.integers(0, len(record_ids), size=len(record_ids))
            exact_means = exact_cube[sampled_indices].mean(axis=(0, 2))
            nll_means = nll_cube[sampled_indices].mean(axis=(0, 2))
            stale_means = stale_cube[sampled_indices].mean(axis=(0, 2))
            exact_draws.append(exact_means[-1] - exact_means[0])
            nll_draws.append(nll_means[-1] - nll_means[0])
            stale_draws.append(stale_means[-1] - stale_means[0])
            exact_slopes.append(np.polyfit(x, exact_means, 1)[0])
            nll_slopes.append(np.polyfit(x, nll_means, 1)[0])
        contrasts[split] = {
            "records": int(len(record_ids)),
            "optimization_seeds": int(subset.seed.nunique()),
            "exact_100_minus_0": point_exact_delta,
            "exact_100_minus_0_ci": interval(np.asarray(exact_draws)),
            "value_nll_100_minus_0": point_nll_delta,
            "value_nll_100_minus_0_ci": interval(np.asarray(nll_draws)),
            "copied_stale_100_minus_0": point_stale_delta,
            "copied_stale_100_minus_0_ci": interval(np.asarray(stale_draws)),
            "exact_linear_slope_per_full_corruption": float(np.polyfit(x, point.exact.to_numpy(), 1)[0]),
            "exact_linear_slope_ci": interval(np.asarray(exact_slopes)),
            "value_nll_linear_slope_per_full_corruption": float(np.polyfit(x, point.value_nll.to_numpy(), 1)[0]),
            "value_nll_linear_slope_ci": interval(np.asarray(nll_slopes)),
            "bootstrap_unit": "record_id; all four corruption conditions and three seeds retained per draw",
            "bootstrap_replicates": args.bootstrap_replicates,
        }
    (output / "contrasts.json").write_text(json.dumps(contrasts, indent=2) + "\n")
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()
