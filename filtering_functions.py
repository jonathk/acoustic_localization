import numpy as np
from scipy import signal
import librosa


def butter_filter(data, cutoff, fs, order=5, filter_type=None):
    """
    :param data: input audio data nx1
    :param cutoff:
    :param fs: nyquist frequency (float)
    :param order: filter order (int)
    :param filter_type: specifies the filter type
    :return y: filtered audio data
    """

    nyq = 0.5 * fs   # Establish nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype=filter_type, analog=False)
    y = signal.lfilter(b, a, data)
    y = np.transpose(y)

    return y


def butter_band(in_data, low_cut, high_cut, fs, order=2, filt_type='bandpass'):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high = high_cut / nyq
    b, a = signal.butter(order, [low, high], filt_type)
    y = signal.filtfilt(b, a, in_data)
    y = np.transpose(y)
    return y


def apply_filter(in_data, fs, cutoff, order=5, filter_type="high", print_info=False):

    # Instantiate helper variables
    n_cols = np.shape(in_data)[1]
    n_rows = np.shape(in_data)[0]
    y_highpass = np.empty((n_rows, 0), dtype="float32")

    # Apply filter iteratively
    for i in range(0, n_cols):
        if print_info:
            print(">> Applying Highpass Filter with cutoff of", str(cutoff), "Hz on Channel: ", i + 1)

        y = in_data[:, i]
        y = butter_filter(y, cutoff, fs, order, filter_type=filter_type).reshape(n_rows, 1)
        y_highpass = np.concatenate((y_highpass, y), axis=1).astype("float32")

    return y_highpass


def apply_band_filters(in_data, fs, f_ranges, order=4, filt_type="bandpass", print_info=False):
    # Instantiate helper variables
    n_cols = np.shape(in_data)[1]
    n_rows = np.shape(in_data)[0]
    y_band = np.zeros((n_rows, n_cols))

    # Apply filters iteratively
    for j in range(0, len(f_ranges)):
        cutoff = f_ranges[j]
        for i in range(0, n_cols):
            if print_info:
                print(">> Applying Bandpass filter of range [" + str(cutoff[0]) + " to",
                      str(cutoff[1]) + "] Hz on Channel: ", i + 1)

            # On j==0, loads in the input data. For the instance that more than 1 interest range is given, then j>0,
            # and rather than loading from the input data, it loads from the previously applied bandpass data
            if j == 0:
                y = in_data[:, i]   # Loads in the current channel of data for the current bandpass f_range
            else:
                y = y_band[:, i]

            y = butter_band(y, cutoff[0], cutoff[1], fs, filt_type=filt_type, order=order)
            y_band[:, i] = y

    return y_band


def decompose(in_data, margin=1):
    D = librosa.stft(in_data)
    H, P = librosa.decompose.hpss(D)

    return H

