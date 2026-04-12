import numpy as np
from config import TRUE_PARAMS, TE
from data_generation import dataset
from varpro import gauss_newton_varpro
from scipy_opt import scipy_varpro_optimize


def reorder_params(C1, C2, T21, T22):
    if T21 > T22:
        return C2, C1, T22, T21
    return C1, C2, T21, T22


def mean(y_data):
    y_data = np.asarray(y_data, dtype=float)
    return np.mean(y_data)


def RMSE(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def run_N_realizations(
    N=100,
    sigma=0.05,
    alpha0=(20.0, 150.0),
    max_iter=1000,
    tol=1e-8,
    damping=1e-3,
):
    # Data to be used for Histogram plot
    true_params = TRUE_PARAMS.copy()
    varpro_results = {"C1": [], "C2": [], "T21": [], "T22": []}
    scipy_results = {"C1": [], "C2": [], "T21": [], "T22": []}

    for seed in range(N):
        noisy_signal = dataset(TE, params=true_params, sigma=sigma, seed=seed)[1]

        # Run VarPro
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

        # Run SciPy
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

    def summarize_results(results_dict, true_params):
        summary = {}
        for key in ["C1", "C2", "T21", "T22"]:
            vals = np.array(results_dict[key], dtype=float)
            summary[key] = {
                "mean": mean(vals),
                "rmse": RMSE(vals, true_params[key]),
            }
        return summary

    def print_summary(method_name, summary):
        print(f"\n==={method_name} Summary ===")
        for key in ["C1", "C2", "T21", "T22"]:
            print(
                f"{key}:mean = {summary[key]['mean']:.6f}, "
                f"RMSE = {summary[key]['rmse']:.6f}"
            )

    varpro_summary = summarize_results(varpro_results, true_params)
    scipy_summary = summarize_results(scipy_results, true_params)

    print_summary("VarPro", varpro_summary)
    print_summary("SciPy", scipy_summary)

    return {
        "varpro_results": varpro_results,
        "scipy_results": scipy_results,
        "varpro_summary": varpro_summary,
        "scipy_summary": scipy_summary,
    }
