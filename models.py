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


def Phi_alpha(TE, T21, T22):
    """
    Build the Matrix Phi(alpha)

    """
    TE = np.asarray(TE)

    col1 = np.exp(-TE / T21)
    col2 = np.exp(-TE / T22)

    return np.column_stack((col1, col2))


def DPhi_alpha_tensor(TE, T21, T22):
    """
    Build DPhi with shape (p,n,m)
    where:
        p = 2 nonlinear parameter
        n = number of data points
        m = 2 linear parameter
    """
    TE = np.asarray(TE)

    exp1 = np.exp(-TE / T21)
    exp2 = np.exp(-TE / T22)

    dcol1_dT21 = (TE / T21**2) * exp1
    dcol2_dT22 = (TE / T21**2) * exp2

    dPhi_dT21 = np.column_stack((dcol1_dT21, np.zeros_like(TE)))
    dPhi_dT22 = np.column_stack((np.zeros_like(TE), dcol2_dT22))

    return np.stack((dPhi_dT21, dPhi_dT22), axis=0)
