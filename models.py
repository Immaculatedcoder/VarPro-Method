import numpy as np


def biexponential(TE, C1, C2, T21, T22):
    """
    Bi-exponential decay model:
        S(TE) = C1*exp(-TE/T21) + C2*exp(-TE/T22)

    Parameters
    ----------
    TE: 1D array or List
        Independent variable
    C1, C2, T21, T22: float
        Model parmeters

    Returns
    -------
    1D array signal values at TE
    """

    TE = np.asarray(TE)
    return C1 * np.exp(-TE / T21) + C2 * np.exp(-TE / T22)
