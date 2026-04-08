def plot_signals(
    TE,
    clean_signal=None,
    noisy_signal=None,
    fitted_signals=None,  # list of (signal, label)
    sigma=None,
    title="Generating Bi-exponential Data",
):
    import matplotlib.pyplot as plt
    import seaborn as sns

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
    plt.show()
