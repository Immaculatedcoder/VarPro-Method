import numpy as np

# True model parameters for data generation
TRUE_PARAMS = {"C1": 0.4, "C2": 0.6, "T21": 30, "T22": 200}

# Echo Times
TE = np.linspace(0, 500, 1000)

# Noise levels
σ = [0.01, 0.05, 0.1, 0.2]
