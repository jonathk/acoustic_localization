import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import shift
import sounddevice as sd
import sys
from pprint import pprint


def audio_callback(indata, frames, time, status, put=True):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, file=sys.stderr)
    q.put(indata[:, 1:5], block=True)
    print("I JUST PUT NEW DATA IN")
    print(time)
    """
    Set the input stream as channels 1:5 based on channel documentation for the Respeaker 4 mic array found here:
                            https://wiki.seeedstudio.com/ReSpeaker-USB-Mic-Array/
    """


def load_device(kind_in='input', print_info=True):
    device_info = sd.query_devices(kind=kind_in)    # Queries for the input device

    rate = device_info['default_samplerate']        # Assigns sample rate to be default of the device
    channels = device_info['max_input_channels']    # Pulls info on max channels (does not equal the number of mics)
    device = device_info['name']                    # simple name of the device

    if print_info:
        print("Device Name: ", device)
        print("Default Sample Rate (Hz): ", rate)
        print("Max Number of Channels: ", channels)

    return device, rate, channels


def format_channel_data(in_data, print_info=False, mock_data=False, plot_bool=False, add_noise=False):
    if mock_data:
        y = in_data
    else:
        y = in_data[:, 1:5]
    n_rows, n_cols = np.shape(y)[0], np.shape(y)[1]

    y1, y2, y3, y4 = np.hsplit(y, n_cols)
    ys = [y1, y2, y3, y4]
    ys = np.squeeze(ys)
    ys = np.transpose(ys)

    if add_noise:
        noise = np.random.normal(min(y1), max(y1), size=(n_rows, n_cols))
        x = np.linspace(0, n_rows, n_rows)
        low_signal = max(y1)*np.transpose(np.sin(2*np.pi*10*x))
        low_sig = np.zeros((n_rows, n_cols))
        low_sig[:, 0], low_sig[:, 1], low_sig[:, 2], low_sig[:, 3] = low_signal, low_signal, low_signal, low_signal
        ys = ys + noise + low_sig

    else:
        pass

    ys = np.float32(ys)

    if plot_bool:
        plt.plot(ys)
        plt.legend(["Ch1", "Ch2", "Ch3", "Ch4"])
        plt.title("Raw Channel Data")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude")
        plt.show()

    if print_info:
        print("Number of Rows: ", n_rows)
        print("Number of Columns: ", n_cols)
        print(">> The shape of the formatted chunk is: ", np.shape(ys))

    return ys, n_rows, n_cols


def basic_cc(in_data, corr_mode='full', corr_method='auto', print_info=False):
    reference_signal = np.transpose(in_data[:, 0])
    n_rows, n_cols = np.shape(in_data)[0], np.shape(in_data)[1]

    lags = []
    for i in range(1, n_cols):
        current_signal = np.transpose(in_data[:, i])
        corr = signal.correlate(current_signal, reference_signal, mode=corr_mode, method=corr_method)
        index = np.argmax(corr) - n_rows
        lags.append(index)
        if print_info:
            print("Lag with maximum correlation between Channels 1 and " + str(i + 1) + " is: " + str(index))

    # Simple guess of where source is coming from
    min_lag = min(lags)
    loc = lags.index(min_lag)
    if min_lag > 0:
        which_mic = 1
    else:
        which_mic = loc + 2

    if print_info:
        print("The Source Is In The Direction Of Mic " + str(which_mic))

    return which_mic


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def determine_bin_harmonics(harmonics, frequencies, print_info=False):
    """
    Simple function to determine the closest bins to the hypothetical harmonic frequencies. The target frequency and
    its harmonics might not align with actual bin values depending on the fft parameters
    :param harmonics: array of harmonic frequencies (typically integer multiples of target frequency)
    :param frequencies: all frequency bins produced by the fft
    :return: bin_harmonics: the actual frequency bins that the ideal harmonic values would correspond to
    """
    if print_info:
        print(">> Converting ideal harmonic frequencies to the closest frequency bin")

    bin_harmonics = []
    for n in harmonics:
        bin_harmonic = find_nearest(frequencies[:, 0], n)
        bin_harmonics.append(bin_harmonic)

    return bin_harmonics


def determine_sufficient_power(frequencies, pxx, harmonics, factor_threshold, n_harmonics_threshold=2,
                               max_power_thresh=1000000, print_info=False):
    """
    This function determines whether there is "sufficient" power in a frequency bin to declare it significant. This
    function ultimately decides whether there are harmonics of the target_frequency present and whether the code has
    detected a potential source.
    :param frequencies: nSamples x nChannels array that corresponds to the frequency values of the power spectra
    :param pxx: nSamples x nChannels array that is the power spectra data
    :param harmonics: 1 x nDesiredHarmonics array. Harmonic frequency bins that we are interested in
    :param factor_threshold: This threshold decides how many times above the expected power necessary for a bin to be
                             considered as hosting a harmonic
    :param n_harmonics_threshold: How many bins need to be found hosting a harmonic to tell the code to continue
    :param max_power_thresh: A threshold that allows the program to continue if it exceeds the power threshold set
    :param print_info: Boolean controlling whether non-essential print statements are utilized
    :return mean_power: Float value that is the mean power of the power spectra
    :return bins_of_note: nHarmonics x nChannels+2 array. Contains the factor of how far above expected power each
                          notable bin is, for each channel. For example, Channel 1 finds that the 320Hz bin is 23x
                          above expected power. Bins_of_note records each channel at that frequency bin. The last
                          column is the frequency of the bin itself, to help keep track of which bins are notable
    :return kill_bool: boolean which allows the code to continue, or if not, kills and another chunk is read in
    """

    kill_bool = True

    n_rows = np.shape(pxx)[0]
    n_cols = np.shape(pxx)[1]

    factors, max_powers = [], []
    index_factors, index_powers = [], []
    for i in range(0, n_cols):
        # Determine some power statistics for the channel
        current_channel = pxx[:, i]
        total_power = np.sum(current_channel)
        expected_power = total_power / n_rows  # Power in a bin if power was distributed evenly across all bins
        # expected_power = 1e-10
        # print("EXPECTED: ", expected_power)

        # Determine which bins have significant power based on expected power and factor_threshold
        indices_factor = np.where(current_channel > factor_threshold * expected_power)[0]
        frequencies_of_note = np.asarray(frequencies[indices_factor])[:, 0]
        index_factors.extend(f for f in indices_factor if f not in index_factors and f in harmonics)
        factors.extend(f for f in frequencies_of_note if f not in factors and f in harmonics)

        # Determine which bins have significant power greater than max_power_threshold
        indices_power = np.where(current_channel > max_power_thresh * expected_power)[0]
        frequencies_of_note_power = np.asarray(frequencies[indices_power])[:, 0]
        index_powers.extend(f for f in indices_power if f not in index_powers and f in harmonics)
        max_powers.extend(f for f in frequencies_of_note_power if f not in max_powers and f in harmonics)

    if len(factors) > n_harmonics_threshold:
        print("in here")
        kill_bool = False

    if len(max_powers) > 0:
        kill_bool = False

    if print_info:
        print("Frequency bins that were determined to have sufficient power: ", factors)
        print("Frequency bins that were determined to exceed the set max power threshold: ", max_powers)

    factors.extend(f for f in max_powers if f not in factors and f in harmonics)
    index_factors.extend(f for f in index_powers if f not in index_factors and f in harmonics)
    bins_of_note = factors
    index_of_note = index_factors

    return index_of_note, bins_of_note, kill_bool

    #     # Move on to the power in each specific bin and how it compares to expected values or thresholds
    #
    #     for f in harmonics:
    #         ind = np.where(frequencies[:, 0] == f)[0][0]
    #         power_in_bin = pxx[ind, i]
    #
    #         power_factor = power_in_bin / mean_power
    #
    #         if print_info:
    #             pass
    #         print("Bin of " + str(f) + "Hz on Channel " + str(i + 1) + " is a factor of " +
    #                   str(round(power_factor, 2)) + " above the expected power")
    #
    #         if power_factor > factor_threshold:
    #             bins_of_note[j, 0] = pxx[ind, 0] / expected_power
    #             bins_of_note[j, 1] = pxx[ind, 1] / expected_power
    #             bins_of_note[j, 2] = pxx[ind, 2] / expected_power
    #             bins_of_note[j, 3] = pxx[ind, 3] / expected_power
    #             bins_of_note[j, 4] = f
    #             j += 1
    #
    # # reduces the array where each row contains all 0 (the f bins where each channel did not find significant power)
    # bins_of_note = bins_of_note[~np.all(bins_of_note == 0, axis=1)]
    #
    # print(">> Finished analyzing the harmonics of the target frequency. Results below:")
    # number_of_harmonics = len(bins_of_note)
    # for i in range(0, number_of_harmonics):
    #     f = bins_of_note[i, 4]
    #     avg_factor = np.mean(bins_of_note[i, 0:4])
    #     max_factor = np.max(bins_of_note[i, 0:4])
    #
    #     print(">> The program has identified a harmonic at " + str(round(f, 2)) + "Hz. Averaged across each channel, "
    #           + str(round(avg_factor, 2)) + "x more power was found in the frequency bin than a normal distribution.")
    #
    #     if number_of_harmonics >= n_harmonics_threshold:
    #         kill_bool = False
    #
    #     if max_factor > max_power_thresh:
    #         kill_bool = False
    #         print(">> The program has identified a harmonic that exceeds the set power threshold. This indicates"
    #               " that a very strong signal is originating in the target frequency or one of the harmonic bins.")
    #
    # return mean_power, bins_of_note, kill_bool


def generate_mock_data(tf, n, amp, n_harmonics, delays):
    x = np.linspace(0, n, n)
    sig = amp * np.sin(2 * np.pi * x * tf)
    for i in range(0, n_harmonics):
        sig = sig + (amp / (i + 2)) * np.sin(2 * np.pi * x * tf*(i+2))

    noise1 = np.random.normal(0, 1, n)
    noise2 = np.random.normal(0, 1, n)
    noise3 = np.random.normal(0, 1, n)
    noise4 = np.random.normal(0, 1, n)
    total_sig1 = sig + noise1
    total_sig2 = sig + noise2
    total_sig3 = sig + noise3
    total_sig4 = sig + noise4

    total_sig1 = shift(total_sig1, delays[0], cval=0)
    total_sig2 = shift(total_sig2, delays[1], cval=0)
    total_sig3 = shift(total_sig3, delays[2], cval=0)
    total_sig4 = shift(total_sig4, delays[3], cval=0)

    y = np.transpose(np.vstack((total_sig1, total_sig2, total_sig3, total_sig4)))
    print(np.shape(y))
    return y


def generate_harmonics(sweep_type, i, j, target_frequencies, n_harmonics=10, n_checks=5,
                       harmonics=0, target_frequency=200):
    print(i)
    print(j)
    if sweep_type == "sweep_slow":
        print("im in")
        if (i + 1) > len(target_frequencies):
            i = 0
        if (j + 1) % n_checks == 0 or j == 0:
            print("I'm in here")
            target_frequency = target_frequencies[i]
            harmonics = np.linspace(target_frequency, target_frequency * n_harmonics, n_harmonics)
            print(harmonics)
            i += 1

    elif sweep_type == "sweep_fast":
        if j == 0:
            target_frequency = target_frequencies[0]
        else:
            target_frequency = target_frequencies[j % len(target_frequencies)]
        harmonics = np.linspace(target_frequency, target_frequency * n_harmonics, n_harmonics)

    else:
        print("Not a valid sweep type")

    return harmonics, i, j, target_frequency
