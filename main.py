from config import TRUE_PARAMS, TE, sigma
from data_generation import dataset
from plotting import plot_signals, plot_histogram
from varpro import gauss_newton_varpro
from scipy_opt import scipy_varpro_optimize
from models import biexponential
from statistical_analysis import run_N_realizations, reorder_params


def main():
    print("True Parameters:")
    for key, value in TRUE_PARAMS.items():
        print(f"{key} = {value}")

    # Initial guess for nonlinear parameters [T21, T22]
    alpha0 = [20.0, 150.0]

    for s in sigma:
        clean_signal, noisy_signal = dataset(
            TE=TE,
            params=TRUE_PARAMS,
            sigma=s,
            seed=None,
        )

        # VarPro fit
        varpro_res = gauss_newton_varpro(
            TE=TE,
            Y=noisy_signal,
            alpha0=alpha0,
            max_iter=1000,
            tol=1e-8,
            damping=0e-4,
            verbose=False,
        )

        T21_v, T22_v = varpro_res["alpha_opt"]
        C1_v, C2_v = varpro_res["a_opt"]

        # SciPy fit
        scipy_res = scipy_varpro_optimize(
            TE=TE,
            Y=noisy_signal,
            alpha0=alpha0,
            verbose=False,
        )

        T21_s, T22_s = scipy_res["alpha_opt"]
        C1_s, C2_s = scipy_res["a_opt"]

        # Optional ordering so smaller T2 comes first
        if T21_v > T22_v:
            T21_v, T22_v = T22_v, T21_v
            C1_v, C2_v = C2_v, C1_v

        if T21_s > T22_s:
            T21_s, T22_s = T22_s, T21_s
            C1_s, C2_s = C2_s, C1_s

        varpro_fit = biexponential(TE, C1_v, C2_v, T21_v, T22_v)
        scipy_fit = biexponential(TE, C1_s, C2_s, T21_s, T22_s)

        print(f"\n=== Results for sigma = {s} ===")
        print("VarPro:")
        print(f"  C1  = {C1_v:.6f}")
        print(f"  C2  = {C2_v:.6f}")
        print(f"  T21 = {T21_v:.6f}")
        print(f"  T22 = {T22_v:.6f}")
        print(f"  Objective = {varpro_res['objective']:.6e}")

        print("SciPy:")
        print(f"  C1  = {C1_s:.6f}")
        print(f"  C2  = {C2_s:.6f}")
        print(f"  T21 = {T21_s:.6f}")
        print(f"  T22 = {T22_s:.6f}")
        print(f"  Objective = {scipy_res['objective']:.6e}")

        plot_signals(
            TE=TE,
            clean_signal=clean_signal,
            noisy_signal=noisy_signal,
            fitted_signals=[
                (varpro_fit, "VarPro fit"),
                (scipy_fit, "SciPy fit"),
            ],
            sigma=s,
            title=f"VarPro vs SciPy Fit (sigma={s})",
        )


def task2():
    sigma = 0.05
    results = run_N_realizations(
        N=10, sigma=sigma, alpha0=(20.0, 150.0), max_iter=1000, tol=1e-8, damping=1e-7
    )
    varpro_results = results["varpro_results"]
    scipy_results = results["scipy_results"]

    for param_name in ["C1", "C2", "T21", "T22"]:
        plot_histogram(
            varpro_data=varpro_results[param_name],
            scipy_data=scipy_results[param_name],
            true_value=TRUE_PARAMS[param_name],
            param_name=param_name,
            sigma=sigma,
            output_dir="results",
        )


if __name__ == "__main__":
    # main()
    task2()
