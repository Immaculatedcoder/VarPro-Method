import numpy as np
from scipy.integrate import solve_ivp


def finite_diff_grad(f, x, eps=1e-8):
    """
    Central-difference gradient of a scalar function f:R^n -> R
    """
    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x)

    for j in range(len(x)):
        e = np.zeros_like(x)
        e[j] = eps
        grad[j] = (f(x + e) - f(x - e)) / (2 * eps)
    return grad


def gradient_proj_ode_solver(
    F,
    gs,
    b,
    x0,
    y0=None,
    grad_F=None,
    grad_g=None,
    eps=1e-8,
    t_span=(0.0, 1000),
    rtol=1e-8,
    atol=1e-10,
    steady_tol=1e-7,
    max_step=np.inf,
):
    """
    Solve
        maximize F(x)
        subject to g_i <= b_i, i=1,...,m
                   x >= 0

        Using the Gradient projection method...
            xdot_j = 0           if x_j = 0 and dL/dx_j < 0
                 = dL/dx_j       otherwise

            ydot_i = 0           if y_i = 0 and dL/dy_i > 0
                   = -dL/dy_i    otherwise

        where
            L(x,y) = F(x) + sum_i y_i(b_i - g_i(x)), y>=0.
    """
    x0 = np.maximum(np.asarray(x0, dtype=float), 0.0)
    b = np.asarray(b, dtype=float)
    m = len(gs)

    if y0 is None:
        y0 = np.zeros(m, dtype=float)
    else:
        y0 = np.maximum(np.asarray(y0, dtype=float), 0.0)

    if len(b) != m:
        raise ValueError("Length of b must equal number of constraints.")

    if grad_g is not None and len(grad_g) != m:
        raise ValueError("Length of grad_g must have same length as g.")

    z0 = np.concatenate([x0, y0])
    n = len(x0)

    def get_grad_F(x_vec):
        if grad_F is not None:
            return np.asarray(grad_F(x_vec), dtype=float)
        return finite_diff_grad(F, x_vec, eps=eps)

    def get_grad_g(i, x_vec):
        if grad_g is not None:
            return np.asarray(grad_g[i](x_vec), dtype=float)
        return finite_diff_grad(gs[i], x_vec, eps)

    def RHS(t, z):
        x = z[:n]
        y = z[n:]

        g_vals = np.array([g(x) for g in gs], dtype=float)

        # dL/dx = grad F - sum_i y_i g_i
        dLdx = get_grad_F(x)  # grad F

        # - sum_i y_i g_i
        for i in range(m):
            dLdx -= y[i] * get_grad_g(i, x)

        # dL/dy = b - g(x)
        dLdy = b - g_vals

        xdot = dLdx.copy()
        ydot = -dLdy.copy()

        # The piecewise boundary conditions
        for j in range(n):
            if x[j] <= 0.0 and dLdx[j] < 0.0:
                xdot[j] = 0.0

        for i in range(m):
            if y[i] <= 0 and dLdy[i] > 0.0:
                ydot[i] = 0.0

        return np.concatenate([xdot, ydot])

    def stop_at_steady(t, z):
        val = np.linalg.norm(RHS(t, z), ord=2) - steady_tol
        return val

    # Events
    stop_at_steady.terminal = True  # the solver stops when the event occurs
    stop_at_steady.direction = 0  # Any crossing

    sol = solve_ivp(
        RHS,
        t_span=t_span,
        y0=z0,
        method="RK45",
        events=stop_at_steady,
        dense_output=True,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )

    z_final = sol.y[:, -1]

    # Enforcing the x>=0, y>=0
    x_star = np.maximum(z_final[:n], 0.0)
    y_star = np.maximum(z_final[n:], 0.0)

    return {
        "x": x_star,
        "y": y_star,
        "objective": F(x_star),
        "g_values": np.array([g(x_star) for g in gs], dtype=float),
        "t": sol.t,
        "z_traj": sol.y.T,
        "nfev": sol.nfev,
    }
