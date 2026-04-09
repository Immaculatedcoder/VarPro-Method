import numpy as np

from models import biexponential


def generate_original_data(TE, params):
    """
    Generate original bi-exponetial data from params without noise

    Parameters
    ----------
    TE: 1D array
    params: dict
        Dictinary with keys: C1, C2, T21, T22.

    Returns:
        1D clean signal
    """
    return biexponential(TE, params["C1"], params["C2"], params["T21"], params["T22"])


def generate_noisy_data(signal, sigma, seed):
    """
    Additive white Gaussian noise to a signal

    Parameters
    ----------
    signal : 1D array.
        Clean Signal
    sigma: float
        Standard deviation of Gaussian noise
    seed: int
        Reprodicity of numbers

    Returns
    -------
    1D array
        Noisy signal
    """

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=len(signal))

    return signal + noise


def dataset(TE, params, sigma, seed=None):
    """
    Generate clean and noisy data

    Returns
    -------
    clean_signal: 1D array
    noisy_signal: 1D array
    """

    clean_signal = generate_original_data(TE, params)
    noisy_signal = generate_noisy_data(clean_signal, sigma, seed)

    return clean_signal, noisy_signal
