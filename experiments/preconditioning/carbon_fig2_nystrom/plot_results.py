"""Plot paper-style training trajectories after both Carbon runs finish."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from run_experiment import RUN_NAMES


BENCHMARK_ENERGY = -37.8450
SMOOTHING_WINDOW = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def moving_average(values: np.ndarray, width: int) -> np.ndarray:
    if values.size < width:
        raise ValueError(f"need at least {width} values, found {values.size}")
    return np.convolve(values, np.ones(width) / width, mode="valid")


def load_metric(run_dir: Path, name: str) -> np.ndarray:
    return np.atleast_1d(np.loadtxt(run_dir / f"{name}.txt")).astype(float)


def main() -> None:
    args = parse_args()
    labels = {"minsr": "minSR", "nystrom": "minSR-Nyström-EMA (r=100)"}
    colors = {"minsr": "tab:blue", "nystrom": "tab:orange"}
    histories: dict[str, dict[str, np.ndarray]] = {}
    for method, run_name in RUN_NAMES.items():
        run_dir = args.output_root / run_name
        histories[method] = {
            "energy": load_metric(run_dir, "energy"),
            "variance": load_metric(run_dir, "variance"),
        }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    archive: dict[str, np.ndarray] = {}
    for method, metrics in histories.items():
        energy = moving_average(metrics["energy"], SMOOTHING_WINDOW)
        variance = moving_average(metrics["variance"], SMOOTHING_WINDOW)
        iterations = np.arange(SMOOTHING_WINDOW, metrics["energy"].size + 1)
        energy_error = np.abs(energy - BENCHMARK_ENERGY)
        axes[0].plot(
            iterations,
            energy_error,
            label=labels[method],
            color=colors[method],
        )
        axes[1].plot(iterations, variance, label=labels[method], color=colors[method])
        archive[f"{method}_iterations"] = iterations
        archive[f"{method}_energy_error"] = energy_error
        archive[f"{method}_variance"] = variance

    axes[0].set_yscale("log")
    axes[0].set_xlabel("Training iteration")
    axes[0].set_ylabel("Absolute energy error (Ha)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Training iteration")
    axes[1].set_ylabel("Local-energy variance (Ha$^2$)")
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()

    artifact_dir = args.output_root / "analysis"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(artifact_dir / "results.png", dpi=200)
    np.savez(artifact_dir / "results.npz", **archive)


if __name__ == "__main__":
    main()
