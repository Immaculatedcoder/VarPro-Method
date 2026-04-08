import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_signals(
    TE,
    clean_signal=None,
    noisy_signal=None,
    fitted_signal=None,
    sigma=None,
    title="Generating Bi-exponential Data",
):
    """
    Plot clean, noisy, and fitted signals
    """

    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("deep")

    plt.figure(figsize=(9, 5), dpi=150)

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
            label=f"Noisy data with sigma={sigma}",
        )

    if fitted_signal is not None:
        plt.plot(
            TE,
            fitted_signal,
            linewidth=2.5,
            linestyle="--",
            color=palette[2],
            label="VarPro fit",
        )

    plt.xlabel("TE", fontsize=12)
    plt.ylabel("S(TE)", fontsize=12)
    plt.title(title, fontsize=14, weight="bold")
    plt.legend(frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
