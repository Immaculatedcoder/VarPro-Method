import numpy as np
from scipy.optimize import minimize

from varpro import reduced_objective, reduced_residual


def scipy_varpro_optimize(
    TE,
    Y,
    alpha0,
    bounds=((1e-6, None), (1e-6, None)),
    method="L-BFGS-B",
    verbose=True,
):
    """
    Optimize nonlinear parameters [T21, T22] using SciPy.

    Parameters
    ----------
    TE : array
    Y : array
    alpha0 : initial guess [T21, T22]
    bounds : bounds for parameters (like fmincon)
    method : optimization method
    verbose : print results

    Returns
    -------
    results : dict
    """

    def objective(alpha):
        T21, T22 = alpha
        return reduced_objective(TE, Y, T21, T22)

    result = minimize(
        objective,
        x0=np.array(alpha0, dtype=float),
        method=method,
        bounds=bounds,
    )

    T21_opt, T22_opt = result.x

    # Recover linear coefficients
    r_opt, a_opt, Phi_opt = reduced_residual(TE, Y, T21_opt, T22_opt)

    if verbose:
        print("\n--- SciPy Optimization Results ---")
        print(f"T21 = {T21_opt:.6f}")
        print(f"T22 = {T22_opt:.6f}")
        print(f"C1  = {a_opt[0]:.6f}")
        print(f"C2  = {a_opt[1]:.6f}")
        print(f"Final objective = {result.fun:.6e}")
        print(f"Success: {result.success}, Message: {result.message}")

    return {
        "alpha_opt": result.x,
        "a_opt": a_opt,
        "objective": result.fun,
        "residual": r_opt,
        "phi_opt": Phi_opt,
        "success": result.success,
        "message": result.message,
    }
