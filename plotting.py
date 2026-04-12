import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_signals(
    TE,
    clean_signal=None,
    noisy_signal=None,
    fitted_signals=None,  # list of (signal, label)
    sigma=None,
    title="Generating Bi-exponential Data",
    output_dir=None,
):

    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("deep")

    plt.figure(figsize=(7, 5), dpi=150)

    if clean_signal is not None:
        plt.plot(
            TE,
            clean_signal,
            linewidth=2.5,
            color=palette[0],
            label="Clean signal",
        )

    if noisy_signal is not None:
        plt.scatter(
            TE,
            noisy_signal,
            s=20,
            color=palette[1],
            alpha=0.6,
            edgecolors="black",
            linewidths=0.2,
            label=f"Noisy (σ={sigma})",
        )

    colors = ["green", "red", "purple", "black"]  # customize here
    linestyles = ["--", "-.", ":", "--"]
    if fitted_signals is not None:
        for i, (signal, label) in enumerate(fitted_signals):
            plt.plot(
                TE,
                signal,
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2.5,
                color=colors[i % len(colors)],
                label=label,
                alpha=0.9,
            )

    plt.xlabel("TE")
    plt.ylabel("S(TE)")
    plt.title(title, weight="bold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"sigma_{sigma}.pdf"
        plt.savefig(filename, bbox_inches="tight")

    plt.show()
    plt.close()


def plot_histogram(
    varpro_data=None,
    scipy_data=None,
    true_value=None,
    param_name=None,
    sigma=None,
    output_dir=None,
):
    sns.set_theme(context="notebook", style="whitegrid")
    plt.figure(figsize=(9, 5), dpi=150)

    varpro_data = np.asarray(varpro_data, dtype=float)
    scipy_data = np.asarray(scipy_data, dtype=float)

    bins = 20

    # ---------
    # Histogram plot
    # ---------

    plt.hist(
        varpro_data,
        bins=bins,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="VarPro",
    )

    plt.hist(
        scipy_data,
        bins=bins,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="SciPy",
    )

    # ----------
    # True Value plot
    # ----------

    plt.axvline(
        true_value,
        color="black",
        linestyle=":",
        linewidth=2,
        label=f"True {param_name}",
    )

    # ------
    # Labels
    # ------

    plt.xlabel(param_name)
    plt.ylabel("Probability Density")
    plt.title(f"Histogram of {param_name}={true_value} (σ={sigma})")
    plt.legend(fontsize=14, loc="best")

    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = output_dir / f"{param_name.lower()}_sigma_{sigma}.pdf"
        plt.savefig(filename, bbox_inches="tight")

    plt.show()
    plt.close()
