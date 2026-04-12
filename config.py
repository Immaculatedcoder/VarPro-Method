import numpy as np

# True model parameters for data generation
TRUE_PARAMS = {"C1": 0.5, "C2": 0.5, "T21": 30, "T22": 200}

# Echo Times
TE = np.linspace(0, 500, 1000)

# Noise levels
sigma = [0.01, 0.05, 0.1, 0.2]
