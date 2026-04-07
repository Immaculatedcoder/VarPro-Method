import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_signals(
    TE,
    clean_signal,
    noisy_signal=None,
    sigma=None,
    title="Generating Bi-exponential Data",
):
    """
    Plot clean and noisy signal

    Parameters
    ----------
    TE: 1D array
    clean_signal: 1D array
    noisy_signal: 1D array
    title: str
    """

    # Seaborn config
    sns.set_theme(style="whitegrid", context="talk")
    palette = sns.color_palette("deep")

    plt.figure(figsize=(9, 5), dpi=100)

    plt.plot(TE, clean_signal, linewidth=2.5, color=palette[0], label="Clean signal")

    if noisy_signal is not None:
        plt.scatter(
            TE,
            noisy_signal,
            s=20,
            color=palette[1],
            alpha=0.6,
            edgecolors="black",
            linewidths=0.2,
            label=f"Noisy data with sigma {sigma}",
        )

    plt.xlabel("TE", fontsize=12)
    plt.ylabel("S(TE)", fontsize=12)
    plt.title(title, fontsize=14, weight="bold")
    plt.legend(frameon=True)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
