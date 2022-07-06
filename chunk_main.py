# Local Import statements
from general_utilities import format_channel_data, determine_sufficient_power, \
    determine_bin_harmonics, generate_mock_data, load_device
from peak_finding_functions import welch_method     # search_peaks, peak_plotter
from localization_functions import custom_delay_sum, bf_delay_sum, bf_music, music2
from plot_functions import plot_data

# Exterior packages
import numpy as np
import sounddevice as sd
from scipy import signal
from arlpy import bf, utils
import time
import matplotlib.pyplot as plt
import xlwt

# Previously used import statements
from pyargus_functions import gen_scanning_vectors, estimate_corr_matrix, DOA_Bartlett
from filtering_functions import apply_filter        # apply_band_filters, decompose
# import queue
# from pprint import pprint
# import scipy as sp
# import scipy.signal
# import arlpy


if __name__ == "__main__":

    # ----------------- Declare the functionalities of the program -----------------
    bf_types = ["custom_ds", "ds", "music", "capon", "esprit", "broad", "mvdr", "music2", "bartlett"]
    bf_type = bf_types[0]           # choose which type of localization method
    plot_bool = False               # generate plots throughout
    use_mock_data = False           # generate mock data rather than true input data
    sweep_fast = True               # sweep each target frequency once, repeatedly
    sweep_slow = False              # sweep one bin multiple times, then move on to next bin, repeatedly
    plot_f_spectrum = True         # plot the frequency spectrum from Welch's Method
    print_info = False              # show the print statements in helper functions
    print_times = False             # print the time it takes to perform each step of the program
    iter_limit_bool = True          # Use an iteration limit to break the program after a certain number of detections
    use_ground_truth = True         # If true, counts how many detections fall within a range around ground truth DoA
    write_data = False              # Determines if data is stored in a excel file

    # ----------------- User-declared variables -----------------
    target_frequencies = [100]      # Fundamental (Target) Frequencies to search for, Hz
    trial = 1
    highpass_cutoff = 70           # Frequency selected to be the cutoff of the highpass filter, Hz
    n_harmonics_thresh = 3          # Num of harmonics needed w/ sufficient power to decide if the signal is harmonic
    filter_order = 8                # Filter order for butterworth filter, keep below 5 because problems with butter
    n_harmonics = 10                # Declare the number of harmonics beyond target that should be investigated
    n_doa_averages = 5              # Number of DOA estimates before an average estimate is yielded
    amplitude = 5                   # Amplitude to use for mock-data signal
    factor_thresh = 150             # How many std above average power for a harmonic to be located (e.g 50x)
    max_power_thresh = 150          # Threshold of std dev for a frequency bin that overrides n_harmonics_thresh require
    bw_desired = 2                  # Frequency bin-width desired for the analysis
    n_averages_welch = 2            # Number of averages for Welch method to take (with overlap)
    n_overlap = 0.5                 # pct overlap expressed as a decimal (e.g, 0.5 will be 50% overlap)
    n_checks = 5                    # How many times to check each bin before proceeding, if using sweep_slow
    iter_limit = 50                 # Number of iterations to perform if iter_limit_bool is set to TRUE
    best_range = 20                 # +/- degrees from ground truth DoA to say if the detection is in the best range
    good_range = 45                 # +/- degrees from ground truth DoA to say if the detection is in the good range
    ground_truth = 180                # actual source azimuth

    # ----------------- Physical constants -----------------
    c = 343                         # sound speed, (m/s)
    d = 0.045                       # distance between mics, (m)

    # ----------------- Physical constants -----------------
    wb = xlwt.Workbook()
    ws = wb.add_sheet('Data')
    ws.write(0, 0, "Chunk")
    ws.write(0, 1, "DoA")

    # ----------------- Load / Establish Device/Sampling Properties -----------------
    time_load_device_in = time.time()
    DEVICE, RATE, _ = load_device(kind_in='input', print_info=True)     # loads the input device
    BLOCK_SIZE = int(RATE/bw_desired*n_averages_welch)                  # Block size to produce desired bin-width
    NYQ = RATE / 2                                                      # Nyquist Frequency
    CHANNELS = 5                                                        # 4 actual channels, see format_channel_data
    RATE = RATE
    N_FFT = BLOCK_SIZE/n_averages_welch                                 # N_FFT value established for Welch Method
    time_load_device_out = time.time()                                  # End time to load device and some constants
    print("Block Size: ", BLOCK_SIZE)
    print("Desired Bin Width (Hz): ", bw_desired)
    print("Sampling Rate (Hz): ", RATE)
    if print_times:
        print("Time taken to load the device and establish constants (s): ", time_load_device_out-time_load_device_in)

    path = 'C:/Users/jonat/Google Drive/Graduate School/Research/RangeTests3/'
    dist = 200
    elev = 5
    filename = str(RATE) + "Hz" + str(dist) + "m" + str(ground_truth) + "deg" + str(bw_desired) + \
               "BWHz" + str(target_frequencies[0]) + "Hz" + str(elev) + "m" + str(trial) + "X8_CONTROL.xls"

    # ----------------- Begin the queue / stream as necessary -----------------
    time_queue_in = time.time()
    if not use_mock_data:                                               # If mock data is not being used, start stream
        stream = sd.InputStream(                                        # Initializes the stream with given parameters
            samplerate=RATE, device=DEVICE,
            channels=CHANNELS, blocksize=BLOCK_SIZE)
        stream.start()                                                  # Starts the stream
    else:
        stream = 0
        print("NO STREAM OBJECT CREATED")
    time_queue_out = time.time()
    if print_times:
        print("Time it took to initialize the queue (s): ", time_queue_out-time_queue_in)

    # ----------------- Begin the main program (iterative DoA search) -----------------
    running_doa = []
    j, i = 0, 0
    n_best_detections, n_good_detections, n_bad_detections, n_no_detections = 0, 0, 0, 0    # sets counter variables
    while True:
        if iter_limit_bool:
            if j >= iter_limit:
                break
        time_start_loop = time.time()
        # Section 1:
        # ----------------- Determine the harmonics to search for this iteration -----------------
        harmonics = 0
        if sweep_slow:                                      # Searches n_checks times for a frequency then moves on
            if (i+1) > len(target_frequencies):             # 1: If the program has searched every frequency in
                i = 0                                       # target_frequencies, set i back to 0 and start again
            if (j+1) % n_checks == 0 or j == 0:             # 2: If the target frequency has been searched n_checks
                target_frequency = target_frequencies[i]    # times, declare the next target frequency
                harmonics = np.linspace(target_frequency, target_frequency * n_harmonics, n_harmonics)
                i += 1
            else:
                target_frequency = 0

        elif sweep_fast:                                    # Searches one time per target_frequency then moves on
            if j == 0:                                      # 1: Declare the first target_frequency as the loop begins
                target_frequency = target_frequencies[0]
            else:                                           # 2: Select the next target frequency
                target_frequency = target_frequencies[j % len(target_frequencies)]
            harmonics = np.linspace(target_frequency, target_frequency * n_harmonics, n_harmonics)

        else:
            harmonics, target_frequency = 0, 0
            print("INVALID SWEEP TYPE SELECTED. SET ONE OF sweep_slow OR sweep_fast TO TRUE")
            break

        print("\n>> ------------ INPUT CHUNK #" + str(j+1) + " ------------")
        print(">> CURRENT TARGET FREQUENCY (HZ): " + str(target_frequency))

        # Section 2:
        # ----------------- Read a data chunk into the stream -----------------
        time_chunk_in = time.time()
        if use_mock_data:                                   # generate mock data if set to true
            y = generate_mock_data(target_frequency, BLOCK_SIZE, amplitude, n_harmonics, delays=[0, 0, 4, 4])
        else:
            y, overflowed = stream.read(frames=BLOCK_SIZE)  # read from the stream if not using mock data
        time_chunk_out = time.time()
        if print_times:
            print("Time it took to load the chunk (s): ", time_chunk_out-time_chunk_in)

        # Section 3:
        # ----------------- Format the chunk data correctly for processing -----------------
        time_format_in = time.time()
        ys, nRows, nCols = format_channel_data(y, print_info=print_info, mock_data=use_mock_data,
                                               plot_bool=plot_bool, add_noise=False)
        time_format_out = time.time()
        if print_times:
            print("Time it took to format the data (s): ", time_format_out-time_format_in)

        # Section 4:
        # ----------------- Apply a highpass filter to each channel of data -----------------
        time_highpass_in = time.time()
        sos = signal.butter(filter_order, highpass_cutoff, btype="hp", fs=RATE, output="sos")
        y_highpass = signal.sosfilt(sos, ys, axis=0)

        if plot_bool:
            plot_data(y_highpass, title="Channel Data after Highpass of " + str(highpass_cutoff) + "Hz",
                      legend=["Ch1", "Ch2", "Ch3", "Ch4"], x_label="Samples", y_label="Amplitude")

        time_highpass_out = time.time()
        if print_times:
            print("Time it took to apply the highpass filter (s): ", time_highpass_out-time_highpass_in)

        # Section 5:
        # ----------------- Apply adaptive band-stop filters to try to filter out propeller noise -----------------
        bpf = [[145-1, 145+1]]
        # y = apply_band_filters(y_highpass, RATE, bpf, order=4, filt_type="bandstop", print_info=False)
        y = y_highpass

        # Section 6:
        # ----------------- Create power spectra via Welch Method -----------------
        time_welch_in = time.time()
        frequencies, pxx = welch_method(y, RATE, nperseg=N_FFT, noverlap=int(N_FFT*n_overlap),
                                        print_info=print_info, plot=plot_f_spectrum)
        time_welch_out = time.time()
        if print_times:
            print("Time it took to perform Welch's method (s): ", time_welch_out-time_welch_in)

        # Section 7:
        # ----------------- Find the closest frequency bin to each harmonic of the target frequency -----------------
        bin_width = frequencies[2][0]-frequencies[1][0]
        print(">> BIN WIDTH (HZ): ", bin_width)
        bin_harmonics = determine_bin_harmonics(harmonics, frequencies, print_info=print_info)

        # Section 8:
        # ----------------- Determine which harmonic frequency bins have above threshold power -----------------
        inds_of_note, bins_of_note, kill_bool = determine_sufficient_power(frequencies, pxx, bin_harmonics,
                                                                           factor_threshold=factor_thresh,
                                                                           n_harmonics_threshold=n_harmonics_thresh,
                                                                           max_power_thresh=max_power_thresh,
                                                                           print_info=print_info)

        # Section 9:
        # ----------------- End this iteration if no harmonics found, or proceed to DoA if they were -----------------
        j += 1
        time_to_detection = time.time()
        if print_times:
            print("Time it took to get to harmonics (s): ", time_to_detection-time_start_loop)
        if kill_bool:
            print(">> Did not identify significant presence of the target frequency or a "
                  "sufficient number of its harmonics. Reading the next chunk in.")
            ws.write(j, 0, j)
            ws.write(j, 1, "-")
            n_no_detections += 1
            continue
        else:
            print(">> Analyzed target frequency harmonics and found " + str(len(bins_of_note)) + " harmonics with "
                  "significant power. Proceeding to next steps.")
            print("These bins are: ", bins_of_note)

        # Section 10:
        # ----------------- Apply a bandpass filter around each located harmonic -----------------
        doa = "-"
        for b in bins_of_note[0:1]:
            low = b - 2
            high = b + 2
            sos = signal.butter(filter_order, low, btype="hp", output="sos", fs=RATE)
            y_filtered = signal.sosfilt(sos, y, axis=0)
            sos = signal.butter(filter_order, high, btype="lp", output="sos", fs=RATE)
            y_filtered2 = signal.sosfilt(sos, y_filtered, axis=0)

            if plot_bool:
                plot_data(y_filtered2, title="Filtered signals from " + str(low) + "Hz to " + str(high) + "Hz",
                          x_label="Samples", y_label="Amplitude", legend=["Ch1", "Ch2", "Ch3", "Ch4"])

            # Perform delay and sum beamforming on the bandpass target frequency data
            doa = 0
            doas = []
            if bf_type == "custom_ds":
                time_custom_ds_in = time.time()
                doa = custom_delay_sum(y_filtered2, RATE, n_search_directions=37, plot_bool=plot_bool)
                doas.append(doa)
                time_custom_ds_out = time.time()
                print("Time it took to perform the custom delay and sum (s): ", time_custom_ds_out-time_custom_ds_in)

            if use_ground_truth:
                good_up, good_down = ground_truth + good_range, ground_truth - good_range
                best_up, best_down = ground_truth + best_range, ground_truth - best_range
                if good_down <= doa < good_up:
                    n_good_detections += 1

                if best_down <= doa <= best_up:
                    n_best_detections += 1

        print("Writing the data to the Excel file")
        ws.write(j, 0, j)
        ws.write(j, 1, doas[0])
        time_doa = time.time()
        print("Time to Detection (s): ", time_doa-time_start_loop)

    wb.save(path+filename)

