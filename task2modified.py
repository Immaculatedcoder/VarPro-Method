# task2.py
# Task from Advisor

"""
1. For N - realization i.e adding noise different times... Infer C1, C2, T21, T22 using VarPro and Scipy
   Plot a histogram of C1, C2, T21, T22.
   Find the mean values of C1, C2, T21, T22.
   Find the RMSC of C1, C2, T21, T22.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import TRUE_PARAMS, TE
from data_generation import dataset
from varpro import gauss_newton_varpro
from scipy_opt import scipy_varpro_optimize
