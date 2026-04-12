# task2.py
# Task from Advisor

"""
1. For N - realization i.e adding noise different times... Infer C1, C2, T21, T22 using VarPro and Scipy
   Plot a histogram of C1, C2, T21, T22.
   Find the mean values of C1, C2, T21, T22.
   Find the RMSC of C1, C2, T21, T22.
"""

# task2.py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import TRUE_PARAMS, TE
from data_generation import dataset
from varpro import gauss_newton_varpro
from scipy_opt import scipy_varpro_optimize


def reorder_params(C1, C2, T21, T22):
    if T21 > T22:
        return C2, C1, T22, T21
    return C1, C2, T21, T22


def rmsc(estimates, true_value):
    estimates = np.asarray(estimates, dtype=float)
    return np.sqrt(np.mean((estimates - true_value) ** 2))


def summarize_results(results_dict, true_params):
    summary = {}
    for key in ["C1", "C2", "T21", "T22"]:
        vals = np.asarray(results_dict[key], dtype=float)
        summary[key] = {
            "mean": np.mean(vals),
            "rmsc": rmsc(vals, true_params[key]),
        }
    return summary


def print_summary(method_name, summary):
    print(f"\n=== {method_name} Summary ===")
    for key in ["C1", "C2", "T21", "T22"]:
        print(
            f"{key}: mean = {summary[key]['mean']:.6f}, "
            f"RMSC = {summary[key]['rmsc']:.6f}"
        )


def save_overlay_histogram_pdf(
    varpro_values,
    scipy_values,
    true_value,
    param_name,
    sigma,
    output_dir,
):
    sns.set_theme(style="whitegrid", context="talk")
    plt.figure(figsize=(8, 5), dpi=300)

    varpro_values = np.asarray(varpro_values)
    scipy_values = np.asarray(scipy_values)

    bins = 20

    # ----------------------------
    # Histogram (more visible)
    # ----------------------------
    plt.hist(
        varpro_values,
        bins=bins,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="VarPro",
    )

    plt.hist(
        scipy_values,
        bins=bins,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="SciPy",
    )

    # ----------------------------

    # ----------------------------
    # True value
    # ----------------------------
    plt.axvline(
        true_value,
        color="black",
        linestyle=":",
        linewidth=3,
        label=f"True {param_name}",
    )

    # ----------------------------
    # Labels
    # ----------------------------
    plt.xlabel(param_name)
    plt.ylabel("Density")
    plt.title(f"Histogram + Gaussian Fit of {param_name} (σ={sigma})")

    plt.legend()
    plt.tight_layout()

    filename = output_dir / f"{param_name.lower()}_sigma_{sigma}.pdf"
    plt.savefig(filename, bbox_inches="tight")
    plt.close()


def run_realization_experiment(
    N=10,
    sigma=0.05,
    alpha0=(20.0, 150.0),
    max_iter=100,
    tol=1e-8,
    damping=1e-3,
    output_dir="task2_results",
):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    true_params = TRUE_PARAMS.copy()

    varpro_results = {"C1": [], "C2": [], "T21": [], "T22": []}
    scipy_results = {"C1": [], "C2": [], "T21": [], "T22": []}

    for seed in range(N):
        _, noisy_signal = dataset(
            TE=TE,
            params=true_params,
            sigma=sigma,
            seed=seed,
        )

        vp = gauss_newton_varpro(
            TE=TE,
            Y=noisy_signal,
            alpha0=alpha0,
            max_iter=max_iter,
            tol=tol,
            damping=damping,
            verbose=False,
        )

        T21_v, T22_v = vp["alpha_opt"]
        C1_v, C2_v = vp["a_opt"]
        C1_v, C2_v, T21_v, T22_v = reorder_params(C1_v, C2_v, T21_v, T22_v)

        varpro_results["C1"].append(C1_v)
        varpro_results["C2"].append(C2_v)
        varpro_results["T21"].append(T21_v)
        varpro_results["T22"].append(T22_v)

        sp = scipy_varpro_optimize(
            TE=TE,
            Y=noisy_signal,
            alpha0=alpha0,
            verbose=False,
        )

        T21_s, T22_s = sp["alpha_opt"]
        C1_s, C2_s = sp["a_opt"]
        C1_s, C2_s, T21_s, T22_s = reorder_params(C1_s, C2_s, T21_s, T22_s)

        scipy_results["C1"].append(C1_s)
        scipy_results["C2"].append(C2_s)
        scipy_results["T21"].append(T21_s)
        scipy_results["T22"].append(T22_s)

        if (seed + 1) % 10 == 0:
            print(f"Completed {seed + 1}/{N} realizations")

    varpro_summary = summarize_results(varpro_results, true_params)
    scipy_summary = summarize_results(scipy_results, true_params)

    print_summary("VarPro", varpro_summary)
    print_summary("SciPy", scipy_summary)

    for param_name in ["C1", "C2", "T21", "T22"]:
        save_overlay_histogram_pdf(
            varpro_values=varpro_results[param_name],
            scipy_values=scipy_results[param_name],
            true_value=true_params[param_name],
            param_name=param_name,
            sigma=sigma,
            output_dir=output_path,
        )

    return {
        "varpro_results": varpro_results,
        "scipy_results": scipy_results,
        "varpro_summary": varpro_summary,
        "scipy_summary": scipy_summary,
    }


if __name__ == "__main__":
    results = run_realization_experiment(
        N=10,
        sigma=0.05,
        alpha0=(20.0, 150.0),
        max_iter=100,
        tol=1e-8,
        damping=1e-3,
        output_dir="task2_results",
    )
