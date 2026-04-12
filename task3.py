# task3.py

# Change the initial guess 5 times... Generate the results for C1, C2, T21, T22

# task3.py
"""
Task:
Change the initial guess 5 times and infer C1, C2, T21, T22
using VarPro and SciPy.
"""

import numpy as np

from config import TRUE_PARAMS, TE
from data_generation import dataset
from varpro import gauss_newton_varpro
from scipy_opt import scipy_varpro_optimize


# -----------------------------
# Helper
# -----------------------------
def reorder_params(C1, C2, T21, T22):
    """
    Ensure T21 <= T22 for consistency
    """
    if T21 > T22:
        return C2, C1, T22, T21
    return C1, C2, T21, T22


# -----------------------------
# Main experiment
# -----------------------------
def run_initial_guess_experiment():
    true_params = TRUE_PARAMS

    # Fix one noisy dataset (important!)
    sigma = 0.05
    _, noisy_signal = dataset(
        TE=TE,
        params=true_params,
        sigma=sigma,
        seed=42,
    )

    # 5 different initial guesses
    initial_guesses = [
        [10.0, 100.0],
        [20.0, 150.0],
        [5.0, 250.0],
        [50.0, 120.0],
        [30.0, 300.0],
    ]

    print("\n=== TRUE PARAMETERS ===")
    for k, v in true_params.items():
        print(f"{k} = {v}")

    print("\n=== RESULTS FOR DIFFERENT INITIAL GUESSES ===")

    for i, alpha0 in enumerate(initial_guesses, start=1):
        print(f"\n--- Initial Guess {i}: {alpha0} ---")

        # -----------------------------
        # VarPro
        # -----------------------------
        vp = gauss_newton_varpro(
            TE=TE,
            Y=noisy_signal,
            alpha0=alpha0,
            max_iter=1000,
            tol=1e-8,
            damping=0e-4,
            verbose=False,
        )

        T21_v, T22_v = vp["alpha_opt"]
        C1_v, C2_v = vp["a_opt"]
        C1_v, C2_v, T21_v, T22_v = reorder_params(C1_v, C2_v, T21_v, T22_v)

        # -----------------------------
        # SciPy
        # -----------------------------
        sp = scipy_varpro_optimize(
            TE=TE,
            Y=noisy_signal,
            alpha0=alpha0,
            verbose=False,
        )

        T21_s, T22_s = sp["alpha_opt"]
        C1_s, C2_s = sp["a_opt"]
        C1_s, C2_s, T21_s, T22_s = reorder_params(C1_s, C2_s, T21_s, T22_s)

        # -----------------------------
        # Print results
        # -----------------------------
        print("VarPro:")
        print(f"  C1={C1_v:.4f}, C2={C2_v:.4f}, " f"T21={T21_v:.2f}, T22={T22_v:.2f}")

        print("SciPy:")
        print(f"  C1={C1_s:.4f}, C2={C2_s:.4f}, " f"T21={T21_s:.2f}, T22={T22_s:.2f}")


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    run_initial_guess_experiment()
