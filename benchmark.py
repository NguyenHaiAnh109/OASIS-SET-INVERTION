"""Compare the accuracy and runtime of OASIS and uniform sampling."""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score

from oasis import BOUNDS as RING_BOUNDS
from oasis import fit_svm, label_ring, oasis
from solve_ellipse import label_ellipse

INITIAL_SAMPLES = 50
TOTAL_SAMPLES = 150
GRID_SIZE = 401
SEEDS = (42, 43, 44)
OUTPUT_DIR = Path(__file__).resolve().parent / "benchmark_results"
CASES = (
    ("Elip", label_ellipse, ((-3.0, 3.0), (-3.0, 3.0))),
    ("Vành khăn", label_ring, RING_BOUNDS),
)
METRICS = {
    "accuracy_pct": "Accuracy (%)",
    "train_seconds": "Lấy mẫu và huấn luyện (giây)",
    "predict_seconds": "Dự đoán toàn bộ lưới (giây)",
}


def benchmark_case(
    name: str,
    label_point: Callable[[np.ndarray], int],
    bounds: tuple[tuple[float, float], ...],
) -> list[dict[str, str | int | float]]:
    """Evaluate both methods on the same fixed grid and sample budget."""
    x = np.linspace(*bounds[0], GRID_SIZE)
    y = np.linspace(*bounds[1], GRID_SIZE)
    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))
    actual = np.array([label_point(point) for point in grid])

    lower, upper = np.asarray(bounds).T
    initial_sigma = 0.05 * float(np.min(upper - lower))
    rows = []
    for seed in SEEDS:
        start = perf_counter()
        active_model, _, _ = oasis(
            label_point, bounds, INITIAL_SAMPLES, TOTAL_SAMPLES, seed
        )
        active_seconds = perf_counter() - start

        # Same seed and uniform sampler share the initial OASIS samples.
        start = perf_counter()
        rng = np.random.default_rng(seed)
        random_points = rng.uniform(lower, upper, size=(TOTAL_SAMPLES, 2))
        random_labels = np.array([label_point(p) for p in random_points])
        random_model = fit_svm(random_points, random_labels, initial_sigma)
        random_seconds = perf_counter() - start

        for method, model, train_seconds in (
            ("OASIS", active_model, active_seconds),
            ("Ngẫu nhiên", random_model, random_seconds),
        ):
            start = perf_counter()
            predicted = model.predict(grid)
            predict_seconds = perf_counter() - start
            rows.append({
                "shape": name,
                "method": method,
                "seed": seed,
                "accuracy_pct": 100 * accuracy_score(actual, predicted),
                "train_seconds": train_seconds,
                "predict_seconds": predict_seconds,
            })
        print(f"{name} | seed {seed} | hoàn tất", flush=True)
    return rows


def save_results(rows: list[dict[str, str | int | float]]) -> None:
    """Save a CSV and a plot with one point per method and seed."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "accuracy_time.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["shape", "method", "seed", *METRICS]
        )
        writer.writeheader()
        writer.writerows(rows)

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, (metric, title) in zip(axes.flat, METRICS.items()):
        sns.stripplot(
            x=[row["shape"] for row in rows],
            y=[row[metric] for row in rows],
            hue=[row["method"] for row in rows],
            dodge=True,
            jitter=False,
            size=7,
            alpha=0.8,
            ax=ax,
        )
        ax.set(title=title, xlabel="", ylabel="")
        ax.get_legend().remove()
        if metric == "train_seconds":
            ax.set_yscale("log")
            ax.set_ylabel("Thang log")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUTPUT_DIR / "accuracy_time.png", dpi=160)
    plt.close(fig)
    print(f"Kết quả: {OUTPUT_DIR}")


def main() -> None:
    """Run the benchmark and save its results."""
    rows = []
    for name, label_point, bounds in CASES:
        rows.extend(benchmark_case(name, label_point, bounds))
    save_results(rows)


if __name__ == "__main__":
    main()
