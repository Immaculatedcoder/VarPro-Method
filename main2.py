import numpy as np
import matplotlib.pyplot as plt
from minmax import gradient_proj_ode_solver


# Objective function
def F(x):
    x1, x2 = x
    return x1 + 2 * x2


def grad_F(x):
    return np.array([1.0, 2.0])


# Constraints
def g1(x):
    x1, x2 = x
    return 3 * x1**2 + x2**2


def grad_g1(x):
    x1, x2 = x
    return np.array([6 * x1, 2 * x2])


def g2(x):
    x1, x2 = x
    return x1 - 8 * x2


def grad_g2(x):
    return np.array([1.0, -8.0])


b = np.array([1.0, -1.0])

result = gradient_proj_ode_solver(
    F=F,
    gs=[g1, g2],
    b=b,
    x0=np.array([1.0, 1.0]),
    y0=np.array([1.0, 1.0]),
    grad_F=grad_F,
    grad_g=[grad_g1, grad_g2],
    t_span=(0.0, 6000.0),
    steady_tol=1e-8,
    rtol=1e-8,
    atol=1e-10,
)

print("x* =", result["x"])
print("y* =", result["y"])
print("F(x*) =", result["objective"])
print("g(x*) =", result["g_values"])


# -----------------------------------
# Extract trajectories
# -----------------------------------
traj = result["z_traj"]
x1_traj = traj[:, 0]
x2_traj = traj[:, 1]
y1_traj = traj[:, 2]
y2_traj = traj[:, 3]

x_star = result["x"]
y_star = result["y"]

# Exact solution for comparison
x_exact = np.array([1 / np.sqrt(39), 6 / np.sqrt(39)])
y_exact = np.array([np.sqrt(39) / 6, 0.0])

# -----------------------------------
# Build grid for feasible region/contours
# -----------------------------------
x1_vals = np.linspace(0.0, 0.7, 500)
x2_vals = np.linspace(0.0, 1.15, 500)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

Z = X1 + 2 * X2

feasible = (
    (3 * X1**2 + X2**2 <= 1.0) & (X1 - 8 * X2 <= -1.0) & (X1 >= 0.0) & (X2 >= 0.0)
)

# Boundary curves
x1_ellipse = np.linspace(0.0, 1 / np.sqrt(3), 500)
x2_ellipse = np.sqrt(np.maximum(0.0, 1 - 3 * x1_ellipse**2))

x1_line = np.linspace(0.0, 0.7, 500)
x2_line = (x1_line + 1.0) / 8.0

# -----------------------------------
# Plot
# -----------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ===== Left plot: x-space =====
ax = axes[0]

# feasible region
ax.contourf(X1, X2, feasible.astype(float), levels=[0.5, 1.5], alpha=0.25)

# objective contours
contours = ax.contour(X1, X2, Z, levels=14, linewidths=1.0)
ax.clabel(contours, inline=True, fontsize=8)

# boundaries
ax.plot(x1_ellipse, x2_ellipse, linewidth=2, label=r"$3x_1^2+x_2^2=1$")
ax.plot(x1_line, x2_line, linewidth=2, label=r"$x_1-8x_2=-1$")

# trajectory
ax.plot(x1_traj, x2_traj, "k-", linewidth=2, label="trajectory")
ax.plot(x1_traj[0], x2_traj[0], "ks", markersize=7, label="start")
ax.plot(x_star[0], x_star[1], "r*", markersize=14, label="numerical optimum")
ax.plot(x_exact[0], x_exact[1], "go", markersize=7, label="exact optimum")

ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
ax.set_title("Feasible region and primal trajectory")
ax.set_xlim(0.0, 0.7)
ax.set_ylim(0.0, 1.15)
ax.grid(True)
ax.legend()

# ===== Right plot: y-space =====
ax = axes[1]
ax.plot(y1_traj, y2_traj, "k-", linewidth=2, label="dual trajectory")
ax.plot(y1_traj[0], y2_traj[0], "ks", markersize=7, label="start")
ax.plot(y_star[0], y_star[1], "r*", markersize=14, label="numerical saddle")
ax.plot(y_exact[0], y_exact[1], "go", markersize=7, label="exact saddle")

ax.set_xlabel(r"$y_1$")
ax.set_ylabel(r"$y_2$")
ax.set_title("Dual trajectory")
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.show()
