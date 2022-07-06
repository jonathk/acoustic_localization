import matplotlib.pyplot as plt
from scipy import signal
import numpy as np


def generate_welch_spectrum(in_data, fs, window_in='hanning', scaling_in='density', npsg_in=None):

    n_rows = np.shape(in_data)[0]
    n_cols = np.shape(in_data)[1]

    frequencies = np.zeros(n_rows, 0)
    power_spectra = np.zeros(n_rows, 0)

    for i in range(0, n_cols):
        f, pxx, = signal.welch(np.transpose(in_data[:, i]), fs, window=window_in, return_onesided=True, scaling=scaling_in, nperseg=12)
        frequencies = np.concatenate((np.transpose(frequencies), f), axis=1)
        power_spectra = np.concatenate((power_spectra, pxx), axis=1)

    return frequencies, power_spectra
