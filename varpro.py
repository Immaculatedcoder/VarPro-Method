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
    TE, Y, alpha0, max_iter=100, tol=1e-8, damping=1e-3, verbose=True
):
    alpha = np.asarray(alpha0, dtype=float).copy()
    history = []

    # Damping bounds to prevent runaway in either direction
    damping_min, damping_max = 1e-12, 1e12

    for k in range(max_iter):
        T21, T22 = alpha

        r, a_hat, Phi = reduced_residual(TE, Y, T21, T22)
        grad = varpro_gradient(TE, Y, T21, T22)
        grad_norm = np.linalg.norm(grad)
        obj = float(r @ r)

        # First-order convergence check (gradient near zero = at a minimum)
        if grad_norm < tol:
            history.append(
                {
                    "iter": k,
                    "alpha": alpha.copy(),
                    "objective": obj,
                    "grad_norm": grad_norm,
                    "step_norm": 0.0,
                }
            )
            if verbose:
                print(f"Converged (||grad||={grad_norm:.3e} < tol).")
            break

        DPhi = DPhi_alpha_tensor(TE, T21, T22)
        J = varpro_jacobian_columns(Y, Phi, DPhi)

        # Marquardt scaling: damping multiplies diag(J^T J), not I.
        # This makes regularization invariant to parameter scaling
        # (important here since T21 ~ 30 and T22 ~ 200).
        JTJ = J.T @ J
        diagJTJ = np.diag(np.diag(JTJ))

        # Try the LM step; if rejected, increase damping and retry
        # *without* recomputing J or grad (inner LM loop).
        accepted = False
        for _ in range(30):
            H = 2 * JTJ + damping * diagJTJ
            try:
                rho = np.linalg.solve(H, -grad)
            except np.linalg.LinAlgError:
                damping = min(damping * 10, damping_max)
                continue

            # Cap step so we never make a parameter non-positive.
            step_scale = 1.0
            alpha_trial = alpha + rho
            if np.any(alpha_trial <= 0):
                # Largest scale that keeps alpha + s*rho > 0
                neg = rho < 0
                if np.any(neg):
                    step_scale = 0.95 * np.min(-alpha[neg] / rho[neg])
                alpha_trial = alpha + step_scale * rho

            obj_trial = reduced_objective(TE, Y, alpha_trial[0], alpha_trial[1])

            if obj_trial < obj:
                accepted = True
                # Successful step: relax damping
                damping = max(damping * 0.5, damping_min)
                break
            else:
                # Rejected: tighten damping and try again
                damping = min(damping * 5, damping_max)
                if damping >= damping_max:
                    break

        step_norm = np.linalg.norm(alpha_trial - alpha) if accepted else 0.0

        history.append(
            {
                "iter": k,
                "alpha": alpha.copy(),
                "objective": obj,
                "grad_norm": grad_norm,
                "step_norm": step_norm,
            }
        )

        if verbose:
            print(
                f"iter={k:2d} | alpha={alpha} | obj={obj:.6e} | "
                f"||grad||={grad_norm:.3e} | step={step_norm:.3e} | "
                f"damping={damping:.2e}"
            )

        if not accepted:
            if verbose:
                print("Line search failed; stopping.")
            break

        alpha = alpha_trial

        if step_norm < tol * (1.0 + np.linalg.norm(alpha)):
            if verbose:
                print("Converged (step size).")
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
