import numpy as np
from scipy.linalg import lstsq
from models import Phi_alpha, DPhi_alpha_tensor


def pseudoinverse(Phi_alpha):
    """Moore-Penrose Matrix inverse"""
    return np.linalg.pinv(Phi_alpha, rcond=1e-12)


def projection_orthogonal(Phi_alpha, Phi_alpha_pinv=None):
    """
    Orthogonal projector onto complement of col(Phi):
        P_perp = I - Phi Phi^+
    """
    n = Phi_alpha.shape[0]
    if Phi_alpha_pinv is None:
        Phi_alpha_pinv = pseudoinverse(Phi_alpha)
    return np.eye(n) - Phi_alpha @ Phi_alpha_pinv


def solve_linear_coefficient(Phi_alpha, Y):
    """
    Solve the linear least squares problem:
        a_hat = argmin_a || Y - Phi_alpha a ||_2^2

    Output
        a -> 1D array (1,2)
            The linear parameter values a = [C1, C2]
    """

    a = pseudoinverse(Phi_alpha) @ Y

    return a


def reduced_residual(TE, Y, T21, T22):
    """
    Compute the reduced residual:
        r(alpha,a) = Y-Phi_alpha(alpha) a
    """

    Phi = Phi_alpha(TE, T21, T22)
    a = solve_linear_coefficient(Phi, Y)
    r = Y - Phi @ a
    return r, a, Phi


def reduced_objective(TE, Y, T21, T22):
    """
    r2(alpha) = ||r(alpha,a)||_2^2
    """

    r = reduced_residual(TE, Y, T21, T22)[0]
    return float(r @ r)


def varpro_gradient(TE, Y, T21, T22):
    """
    (1/2 grad r2)
    """

    Y = np.asarray(Y, dtype=float).reshape(-1)
    Phi = Phi_alpha(TE, T21, T22)
    DPhi = DPhi_alpha_tensor(TE, T21, T22)

    Phi_alpha_pinv = pseudoinverse(Phi)
    P_perp = projection_orthogonal(Phi, Phi_alpha_pinv)

    a_hat = Phi_alpha_pinv @ Y

    p = DPhi.shape[0]
    grad = np.empty(p, dtype=float)

    for k in range(p):
        v_k = DPhi[k] @ a_hat
        grad[k] = -Y @ (P_perp @ v_k)

    return 2 * grad


def varpro_jacobian_columns(Y, Phi, DPhi):
    Phi_pinv = pseudoinverse(Phi)
    P_perp = projection_orthogonal(Phi)
    a_hat = Phi_pinv @ Y

    n = Phi.shape[0]
    p = DPhi.shape[0]
    J = np.zeros((n, p), dtype=float)

    for k in range(p):
        J[:, k] = P_perp @ (DPhi[k] @ a_hat)

    return J


def varpro_hessian(TE, Y, T21, T22):
    Phi = Phi_alpha(TE, T21, T22)
    DPhi = DPhi_alpha_tensor(TE, T21, T22)
    J = varpro_jacobian_columns(Y, Phi, DPhi)
    return 2 * (J.T @ J)


def gauss_newton_varpro(
    TE, Y, alpha0, max_iter=1000, tol=1e-8, damping=0, verbose=True
):
    alpha = np.asarray(alpha0, dtype=float).copy()
    history = []

    for k in range(max_iter):
        T21, T22 = alpha

        grad = varpro_gradient(TE, Y, T21, T22)
        H = varpro_hessian(TE, Y, T21, T22) + damping * np.eye(2)

        obj = reduced_objective(TE, Y, T21, T22)

        try:
            rho = np.linalg.solve(H, -grad)
        except np.linalg.LinAlgError:
            rho = -np.linalg.pinv(H) @ grad

        # Backtracking line search
        step_scale = 1.0
        accepted = False

        for _ in range(20):
            alpha_trial = alpha + step_scale * rho
            alpha_trial = np.maximum(alpha_trial, 1e-10)

            obj_trial = reduced_objective(TE, Y, alpha_trial[0], alpha_trial[1])

            if obj_trial < obj:
                accepted = True
                break

            step_scale *= 0.5

        step_norm = np.linalg.norm(step_scale * rho)

        history.append(
            {
                "iter": k,
                "alpha": alpha.copy(),
                "objective": obj,
                "grad_norm": np.linalg.norm(grad),
                "step_norm": step_norm,
            }
        )

        if verbose:
            print(
                f"iter={k:2d} | "
                f"alpha={alpha} | "
                f"obj={obj:.6e} | "
                f"||grad||={np.linalg.norm(grad):.3e} | "
                f"step={step_norm:.3e}"
            )

        if not accepted:
            if verbose:
                print(
                    "Step did not reduce objective even after backtracking. Stopping."
                )
            break

        alpha = alpha_trial

        if step_norm < tol:
            if verbose:
                print("Converged based on step norm.")
            break

    T21_opt, T22_opt = alpha
    r_opt, a_opt, Phi_opt = reduced_residual(TE, Y, T21_opt, T22_opt)
    obj_opt = float(r_opt @ r_opt)

    return {
        "alpha_opt": alpha,
        "a_opt": a_opt,
        "residual": r_opt,
        "objective": obj_opt,
        "phi_opt": Phi_opt,
        "history": history,
    }
