import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from pprint import pprint


def welch_method(in_data, fs, window='hann', nperseg=256, noverlap=128, print_info=True, plot=False):
    """
    This function uses Welch's Method to compute the power spectra of the time-series chunk.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
    https://stackoverflow.com/questions/44268488/scipy-signal-spectrogram-nfft-parameter
    https://dsp.stackexchange.com/questions/81640/trying-to-understand-the-nperseg-effect-of-welch-method
    https://www.spectraplus.com/DT_help/fft_size.htm#:~:text=The%20frequency%20resolution%20of%20each,but%20take%20longer%20to%20compute.

    :param in_data: (m Samples x n Channels) numpy array of time-data from the mic-array
    :param fs: Sampling Frequency, Hz
    :param window: Type of windowing that the method will use. Defaults to Hanning
    :param nperseg: Number of samples in each fft segment. Dictates the amount of averaging that will occur
    :param noverlap: Number of overlap between each fft segment. A good input is typically 50% (half of nperseg)
    :param print_info: Boolean that determines if print statements are used
    :param plot: Boolean that determines if plots are generated
    :return frequencies: mxn array of frequencies that correspond to the frequency (power) spectra points
    :return power_spectra: mxn array of frequency (power) spectra points
    """

    n_rows = np.shape(in_data)[0]
    n_cols = np.shape(in_data)[1]

    frequencies = []
    power_spectra = []

    for i in range(0, n_cols):
        ch_current = in_data[:, i]
        f, pxx = signal.welch(ch_current, fs, nperseg=nperseg, noverlap=noverlap, window=window, scaling='density')

        frequencies.append(f)
        power_spectra.append(pxx)

    frequencies = np.transpose(frequencies)
    power_spectra = np.transpose(power_spectra)

    if plot:
        plt.semilogx(frequencies, power_spectra)
        plt.title("Power Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Power (V**2)")
        plt.legend(["Ch1", "Ch2", "Ch3", "Ch4"])
        plt.show()

    if print_info:
        print(">> Computing the Power Spectra by Welch's Method")

    return frequencies, power_spectra


def search_power_spectra(in_f, in_pxx, print_info=False):
    """
    :param in_f: mxn input array of frequencies that correspond to the input power (x_data)
    :param in_pxx: mxn input array of power spectra points (y_data)
    :param print_info: boolean that controls if print statements are used
    :return:
    """
    n_cols = np.shape(in_pxx)[1]
    for i in range(0, n_cols):
        chf = in_f[:, i]
        chp = in_pxx[:, i]

        max_index = np.argmax(chp)
        if print_info:
            print("Power Spectra Maximum at Frequency of:", chf[max_index])


def search_peaks(in_data, f, h=0, dist_in=1, prom=None, wid=None, print_info=True):
    """
    This function uses the find_peaks function of the signal module of the scipy package. It can be tuned to find
    peaks of an array of data based on certain conditions, such as height, prominence, width, etc. Documentation can
    be found here: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html

    A guide can be found here: https://blog.finxter.com/python-scipy-signal-find_peaks/

    :param in_data: nSamples x nChannels Array of audio data to find peaks
    :param f: nSamples x nChannels Array of frequencies related to the input data, for printing Hz peaks
    :param h: float of the minimum height value of a peak, normally based on sigma above mean
    :param dist_in: int of how many samples peaks must be separated by
    :param prom: prominence of the peaks
    :param wid: width of the peaks
    :param print_info: boolean to determine if information is printed to user
    :return pks_out: array of peaks found
    """

    if print_info:
        print(">> Finding the peaks of the provided spectra...")

    num_channels = np.shape(in_data)[1]

    pks_out = np.zeros((20, num_channels))
    f_out = np.zeros((20, num_channels))
    max_length = 0

    for i in range(0, num_channels):
        j = 0
        current_channel = in_data[:, i]
        pks = signal.find_peaks(current_channel,
                                height=h,
                                distance=dist_in,
                                prominence=prom,
                                width=wid
                                )

        peak_heights = pks[1]['peak_heights']   # find the height values associated with the pks
        indices = pks[0]                        # find the frequency indices those pks occur at
        for ind in indices:
            f_value = f[ind, i]                 # frequency value is the index and channel of the array
            f_out[j, i] = f_value               # append it to the f_out array
            pks_out[j, i] = peak_heights[j]     # same for pks_out, assign it the proper height value
            j += 1

            if len(indices) > max_length:       # keeping track of the max peaks found across all channels
                max_length = len(indices)       # this allows reducing of the pks_out and f_out arrays at the end

        if print_info:                          # prints each peak for the channel to terminal if desired
            for n, j in enumerate(pks[0]):
                print("Peak Found for Channel ", i+1, " at: ", f[j][i], " Hz")

    f_out = f_out[~np.all(f_out == 0, axis=1)]  # reduces the array where each row contains 0 (doesn't lose any peaks)
    pks_out = pks_out[~np.all(pks_out == 0, axis=1)]

    return pks_out, f_out


def peak_plotter(pks, pks_f, pxx, pxx_f, tf):
    """
    :param pks: mxn array of peak values (y_data point values), these are points of located peaks
    :param pks_f: mxn array of frequencies that correspond to the points of the peaks
    :param pxx: mxn array of power spectra y data
    :param pxx_f: frequency array that corresponds to the pxx data
    :return:
    """
    markers_for_plt = []
    styles = ['xr', 'ob', 'vg', 'xk']
    lines = plt.semilogx(pxx_f, pxx)

    n_rows = np.shape(pks)[0]
    n_cols = np.shape(pks)[1]

    # plot peak locations as markers
    for i in range(0, n_cols):
        for j in range(0, n_rows):
            pk_height = pks[j, i]
            pk_f = pks_f[j, i]
            pts = plt.plot(pk_f, pk_height, styles[i])
            if j == 0:
                markers_for_plt.append(pts)

    # markers_for_plt = np.array(markers_for_plt)
    markers_for_plt = np.reshape(markers_for_plt, (1, 4))
    markers_for_plt = markers_for_plt[0]
    markers_for_plt = np.transpose(markers_for_plt)
    lines = np.array(lines)
    plts = np.hstack((lines, markers_for_plt))
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power Spectral Density (V**2/Hz)")
    plt.legend(plts, ['Channel 1', 'Channel 2', 'Channel 3', 'Channel 4',
                      'Ch1 Peak', 'Ch2 Peak', 'Ch3 Peak', 'Ch4 Peak'])
    plt.title("Located Peaks\n" + str(tf) + "Hz Target Frequency")
    plt.show()


def peak_and_harmonic_comparisons(pks, harmonics):
    pass