from config import TRUE_PARAMS, TE, sigma
from data_generation import dataset
from plotting import plot_signals
from varpro import gauss_newton_varpro
from models import biexponential


def main():
    print("True Parameters:")
    for key, value in TRUE_PARAMS.items():
        print(f"{key} = {value}")

    # Initial guess for nonlinear parameters [T21, T22]
    alpha0 = [15.0, 220.0]

    for s in sigma:
        clean_signal, noisy_signal = dataset(
            TE=TE,
            params=TRUE_PARAMS,
            sigma=s,
            seed=42,
        )

        results = gauss_newton_varpro(
            TE=TE,
            Y=noisy_signal,
            alpha0=alpha0,
            max_iter=50,
            tol=1e-8,
            damping=1e-10,
            verbose=True,
        )

        T21_opt, T22_opt = results["alpha_opt"]
        C1_opt, C2_opt = results["a_opt"]

        # Optional ordering so smaller T2 comes first
        if T21_opt > T22_opt:
            T21_opt, T22_opt = T22_opt, T21_opt
            C1_opt, C2_opt = C2_opt, C1_opt

        # Make sure this argument order matches your models.py
        fitted_signal = biexponential(TE, C1_opt, C2_opt, T21_opt, T22_opt)

        print(f"\nResults for sigma = {s}")
        print(f"Estimated C1  = {C1_opt:.6f}")
        print(f"Estimated C2  = {C2_opt:.6f}")
        print(f"Estimated T21 = {T21_opt:.6f}")
        print(f"Estimated T22 = {T22_opt:.6f}")
        print(f"Final objective = {results['objective']:.6e}")

        plot_signals(
            TE=TE,
            clean_signal=clean_signal,
            noisy_signal=noisy_signal,
            fitted_signal=fitted_signal,
            sigma=s,
            title=f"VarPro Fit (sigma={s})",
        )


if __name__ == "__main__":
    main()
