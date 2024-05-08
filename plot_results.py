from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes


def set_font_size(font_size: int) -> None:
    """
    References:
        https://stackoverflow.com/a/39566040
    """
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.titlesize": font_size,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size,
            "ytick.labelsize": font_size,
            "legend.fontsize": font_size,
            "figure.titlesize": font_size,
        }
    )


def plot(axes: Axes, results: pd.DataFrame, metric: str, label: str = None) -> None:
    axes.plot(results["n_labels"], results[f"test_{metric}_mean"], label=label)
    axes.fill_between(
        results["n_labels"],
        results[f"test_{metric}_mean"] + results[f"test_{metric}_sem"],
        results[f"test_{metric}_mean"] - results[f"test_{metric}_sem"],
        alpha=0.3,
    )
    axes.grid(visible=True, axis="y")


def main() -> None:
    results_dir = Path("results")

    encoders = {
        "barlow": "Barlow Twins",
        "byol": "BYOL",
        "deepcluster": "DeepCluster v2",
        "dino": "DINO",
        "mocov2": "MoCo v2",
        "mocov3": "MoCo v3",
        "nnclr": "NNCLR",
        "ressl": "ReSSL",
        "simclr": "SimCLR",
        "swav": "SwAV",
        "vibcreg": "VIbCReg",
        "vicreg": "VICReg",
        "wmse": "W-MSE",
    }

    results = {}

    for dataset in ("cifar10", "cifar100"):
        results[dataset] = {}

        for encoder in encoders:
            for i, filepath in enumerate((results_dir / dataset / encoder).glob("*.csv")):
                _, seed_str = filepath.stem.split("_")

                column_mapper = {
                    "test_acc": f"test_acc_{seed_str}",
                    "test_loglik": f"test_loglik_{seed_str}",
                }

                run_results = pd.read_csv(filepath).rename(columns=column_mapper)

                if i == 0:
                    _results = run_results
                else:
                    _results = _results.merge(run_results, on="n_labels")

            for metric in ("acc", "loglik"):
                _results[f"test_{metric}_mean"] = _results.filter(regex=metric).mean(axis=1)
                _results[f"test_{metric}_sem"] = _results.filter(regex=metric).sem(axis=1)

            _results[_results.filter(regex="acc").columns] *= 100

            results[dataset][encoder] = _results

    set_font_size(11)

    figure, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 6))

    for i, metric in enumerate(("acc", "loglik")):
        y_label = "Test accuracy (%)" if metric == "acc" else "Test expected log likelihood"

        for encoder in encoders:
            plot(axes[i, 0], results["cifar10"][encoder], metric, label=encoders[encoder])
            plot(axes[i, 1], results["cifar100"][encoder], metric, label=encoders[encoder])

        axes[i, 0].set(title="CIFAR-10", xlabel="Number of labels", ylabel=y_label)
        axes[i, 1].set(title="CIFAR-100", xlabel="Number of labels")

    legend_handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(legend_handles, legend_labels, borderpad=0.5, bbox_to_anchor=(1.23, 0.92))

    figure.tight_layout()
    figure.savefig(results_dir / "plot.svg", bbox_inches="tight")


if __name__ == "__main__":
    main()
