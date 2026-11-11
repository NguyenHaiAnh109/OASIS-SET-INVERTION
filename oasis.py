"""Approximate a preimage with OASIS and plot a 2D example.

Python 3.9+. Install numpy, scipy, scikit-learn, matplotlib and
seaborn, then run: python oasis.py.
"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.optimize import brentq, minimize
from sklearn.svm import SVC

BOUNDS = ((-2.0, 2.0), (-2.0, 2.0))
INITIAL_SAMPLES = 50
TOTAL_SAMPLES = 150
RANDOM_SEED = 42
SVM_C = 1000.0
MAX_ATTEMPTS = 20
BOUNDARY_TOL = 1e-6


def fit_svm(
    points: np.ndarray, labels: np.ndarray, initial_sigma: float
) -> SVC:
    """Try RBF widths, then increase C for close opposite labels."""
    candidates = [
        (initial_sigma * 1.25**step, SVM_C) for step in range(12)
    ]
    candidates.extend(
        (initial_sigma, SVM_C * factor) for factor in (10, 100, 1000)
    )
    for sigma, c in candidates:
        # The paper uses a width; sklearn uses the exponent coefficient.
        gamma = 1.0 / (2.0 * sigma**2)
        model = SVC(kernel="rbf", C=c, gamma=gamma, tol=1e-5)
        model.fit(points, labels)
        if np.array_equal(model.predict(points), labels):
            return model

    raise RuntimeError("No tested SVM settings separated the training points.")


def sample_boundary(
    model: SVC,
    points: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Start SLSQP from a point already on the decision boundary."""
    lower, upper = np.asarray(bounds, dtype=float).T
    scores = model.decision_function(points)
    positive = points[scores > 0]
    negative = points[scores < 0]
    if not len(positive) or not len(negative):
        raise RuntimeError("Need samples on both sides of the SVM boundary.")

    def decision(point: np.ndarray) -> float:
        return float(model.decision_function(point.reshape(1, -1))[0])

    def is_new(point: np.ndarray) -> bool:
        return bool(np.all(np.linalg.norm(points - point, axis=1) >= 1e-8))

    for _ in range(MAX_ATTEMPTS):
        anchor = rng.uniform(lower, upper)
        left = negative[np.argmin(np.linalg.norm(negative - anchor, axis=1))]
        right = positive[np.argmin(np.linalg.norm(positive - anchor, axis=1))]
        direction = right - left

        # Opposite signs at the endpoints bracket a zero of the SVM.
        t = brentq(lambda t: decision(left + t * direction), 0.0, 1.0)
        start = left + t * direction
        result = minimize(
            lambda point: np.sum((point - anchor) ** 2),
            x0=start,
            method="SLSQP",
            bounds=bounds,
            constraints={"type": "eq", "fun": decision},
            options={"ftol": 1e-9},
        )
        if (
            result.success
            and abs(decision(result.x)) <= BOUNDARY_TOL
            and is_new(result.x)
        ):
            return result.x
        if abs(decision(start)) <= BOUNDARY_TOL and is_new(start):
            return start

    raise RuntimeError("Could not find a distinct point on the SVM boundary.")


def oasis(
    label_point: Callable[[np.ndarray], int],
    bounds: tuple[tuple[float, float], ...],
    n_initial: int,
    n_total: int,
    seed: int = RANDOM_SEED,
) -> tuple[SVC, np.ndarray, np.ndarray]:
    """Return the final SVM, queried points and their true labels.

    label_point returns +1 when F(x) belongs to U and -1 otherwise.
    The first n_initial points are random; later points are active.
    """
    if not 2 <= n_initial <= n_total:
        raise ValueError("Require 2 <= n_initial <= n_total.")

    lower, upper = np.asarray(bounds, dtype=float).T
    if np.any(lower >= upper):
        raise ValueError("Each lower bound must be below its upper bound.")

    rng = np.random.default_rng(seed)
    points = rng.uniform(lower, upper, size=(n_initial, len(bounds)))
    labels = np.array([label_point(point) for point in points])
    if set(labels) != {-1, 1}:
        raise ValueError("Initial points must contain both labels: -1 and 1.")

    # Start with a small width relative to the search box.
    initial_sigma = 0.05 * float(np.min(upper - lower))
    model = fit_svm(points, labels, initial_sigma)

    while len(points) < n_total:
        new_point = sample_boundary(model, points, bounds, rng)
        new_label = label_point(new_point)
        if new_label not in (-1, 1):
            raise ValueError("label_point must return -1 or 1.")

        points = np.vstack((points, new_point))
        labels = np.append(labels, new_label)
        model = fit_svm(points, labels, initial_sigma)

    return model, points, labels


def label_ring(point: np.ndarray) -> int:
    """Label a point in the ring defined by U = [1, 2]."""
    value = float(point @ point)
    return 1 if 1.0 <= value <= 2.0 else -1


def plot_ring(
    model: SVC, points: np.ndarray, labels: np.ndarray, n_initial: int
) -> None:
    """Save the samples, learned boundary, and true boundary as a PNG."""

    from pathlib import Path

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 7))

    # Generate grid
    x = np.linspace(*BOUNDS[0], 220)
    y = np.linspace(*BOUNDS[1], 220)

    xx, yy = np.meshgrid(x, y)
    grid = np.column_stack((xx.ravel(), yy.ravel()))

    # Calculate SVM decision scores
    scores = model.decision_function(grid).reshape(xx.shape)

    # Classify sample types
    sample_types = np.where(
        np.arange(len(points)) < n_initial,
        "Random",
        "Active",
    )

    # Classify membership
    membership = np.where(
        labels == 1,
        "Inside",
        "Outside",
    )

    # Plot samples
    sns.scatterplot(
        x=points[:, 0],
        y=points[:, 1],
        hue=membership,
        style=sample_types,
        palette={
            "Inside": "#0072B2",
            "Outside": "#D55E00",
        },
        ax=ax,
    )

    # Learned decision boundary (solid black)
    ax.contour(
        xx,
        yy,
        scores,
        levels=[0],
        colors="black",
        linewidths=1.8,
    )

    # True decision boundaries (dashed gray)
    ax.contour(
        xx,
        yy,
        xx**2 + yy**2,
        levels=[1, 2],
        colors="gray",
        linestyles="dashed",
        linewidths=1.5,
    )

    # Plot settings
    ax.set(
        title="OASIS Ring | Solid: Learned, Dashed: True",
        xlabel=r"$x_1$",
        ylabel=r"$x_2$",
        xlim=BOUNDS[0],
        ylim=BOUNDS[1],
        aspect="equal",
    )

    # Adjust figure layout
    fig.subplots_adjust(
        left=0.12,
        right=0.95,
        bottom=0.10,
        top=0.92,
    )

    # Save figure
    output_path = (
        Path(__file__).resolve().parent / "oasis_result.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    # Release memory
    plt.close(fig)


def main() -> None:
    """Run OASIS on the paper's 2D ring example."""
    try:
        model, points, labels = oasis(
            label_ring, BOUNDS, INITIAL_SAMPLES, TOTAL_SAMPLES
        )
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"OASIS: {error}") from error

    print(f"Random samples: {INITIAL_SAMPLES}")
    print(f"Active samples: {len(points) - INITIAL_SAMPLES}")
    plot_ring(model, points, labels, INITIAL_SAMPLES)


if __name__ == "__main__":
    main()
