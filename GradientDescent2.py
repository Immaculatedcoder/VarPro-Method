import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# RHS function
def RHS(t, z):
    x1, x2, y1, y2 = z
    dz = np.zeros(4)

    dLdx1 = -16*x1 + 12*x2 - 50 - y1 - 16*y2*x1
    dLdx2 = -20*x2 + 12*x1 + 80 - y1 - 2*x2*y2
    dLdy1 = 1 - x1 - x2
    dLdy2 = 2 - 8*x1**2 - x2**2

    dz[0] = 0 if (dLdx1 < 0 and x1 < 0) else dLdx1
    dz[1] = 0 if (dLdx2 < 0 and x2 < 0) else dLdx2
    dz[2] = 0 if (dLdy1 > 0 and y1 < 0) else -dLdy1
    dz[3] = 0 if (dLdy2 > 0 and y2 < 0) else -dLdy2

    return dz

# Event function to stop at steady state
def stop_at_steady(t, z):
    tol = 0.1
    dz = RHS(t, z)
    value = np.linalg.norm(dz) - tol
    return value

stop_at_steady.terminal = True
stop_at_steady.direction = 0

# Initial conditions and time span
z0 = [1, 1, 1, 1]
tspan = (0, 6000)

# Solve ODE
sol = solve_ivp(RHS, tspan, z0, method='RK45', events=stop_at_steady, dense_output=True)

t = sol.t
z = sol.y.T  # transpose to match MATLAB column-wise indexing

x1, x2, y1, y2 = z[:,0], z[:,1], z[:,2], z[:,3]

print(f'Final Values: ({x1[-1]:.6f}, {x2[-1]:.6f}, {y1[-1]:.6f}, {y2[-1]:.6f}) at t = {t[-1]:.6f}')

# Plotting
plt.figure(figsize=(12,5))

# x1 vs x2
plt.subplot(1,2,1)
plt.plot(x1, x2, 'k-', linewidth=2)
plt.plot(x1[0], x2[0], 'rs', linewidth=2, label='Starting Point')
plt.plot(x1[-1], x2[-1], 'rx', linewidth=2, label='Saddle Point')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True)

# y1 vs y2
plt.subplot(1,2,2)
plt.plot(y1, y2, 'k-', linewidth=2)
plt.plot(y1[0], y2[0], 'rs', linewidth=2, label='Starting Point')
plt.plot(y1[-1], y2[-1], 'rx', linewidth=2, label='Saddle Point')
plt.xlabel('y1')
plt.ylabel('y2')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()