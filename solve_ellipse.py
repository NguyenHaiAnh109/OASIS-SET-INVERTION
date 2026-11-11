"""Use OASIS to learn the boundary of a two-dimensional ellipse."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.svm import SVC

from oasis import oasis

BOUNDS = ((-3.0, 3.0), (-3.0, 3.0))
INITIAL_SAMPLES = 100
TOTAL_SAMPLES = 150


def label_ellipse(point: np.ndarray) -> int:
    """Return +1 inside or on the ellipse and -1 outside."""
    x1, x2 = point
    return 1 if x1**2 + 4 * x2**2 <= 1.0 else -1


def plot_ellipse(
    model: SVC, points: np.ndarray, labels: np.ndarray
) -> None:
    """Save the samples, learned boundary, and true ellipse boundary."""

    from pathlib import Path

    sns.set_theme(style="whitegrid")

    fig, ax = plt.subplots(figsize=(7, 7))

    # Generate grid
    x1 = np.linspace(*BOUNDS[0], 220)
    x2 = np.linspace(*BOUNDS[1], 220)

    xx, yy = np.meshgrid(x1, x2)
    grid = np.column_stack((xx.ravel(), yy.ravel()))

    # Calculate SVM decision scores
    scores = model.decision_function(grid).reshape(xx.shape)

    # Classify membership
    membership = np.where(
        labels == 1,
        "Trong",
        "Ngoài",
    )

    # Classify sample types
    sample_types = np.where(
        np.arange(len(points)) < INITIAL_SAMPLES,
        "Ngẫu nhiên",
        "Chủ động",
    )

    # Plot samples
    sns.scatterplot(
        x=points[:, 0],
        y=points[:, 1],
        hue=membership,
        style=sample_types,
        palette={
            "Trong": "#0072B2",
            "Ngoài": "#D55E00",
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

    # True ellipse boundary (dashed gray)
    ax.contour(
        xx,
        yy,
        xx**2 + 4 * yy**2,
        levels=[1],
        colors="gray",
        linestyles="dashed",
        linewidths=1.5,
    )

    # Plot settings
    ax.set(
        title="OASIS | Predicted Boundary (Solid) and True Boundary (Dashed)",
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
        Path(__file__).resolve().parent / "ellipse_result.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    # Release memory
    plt.close(fig)


def main() -> None:
    """Run OASIS on the ellipse example and display the result."""
    try:
        model, points, labels = oasis(
            label_ellipse, BOUNDS, INITIAL_SAMPLES, TOTAL_SAMPLES
        )
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"OASIS: {error}") from error

    print(f"Random samples: {INITIAL_SAMPLES}")
    print(f"Active samples: {len(points) - INITIAL_SAMPLES}")
    plot_ellipse(model, points, labels)


if __name__ == "__main__":
    main()
