import numpy as np
import numpy.linalg as la
from scipy.ndimage.interpolation import shift
from pyargus_functions import gen_scanning_vectors, estimate_corr_matrix, doa_music, doa_capon
from peak_finding_functions import welch_method
import matplotlib.pyplot as plt
from arlpy import bf
import librosa
from librosa import display
from pprint import pprint


def covariance(x):
    """Compute array covariance matrix.
    :param x: narrowband complex timeseries data for multiple sensors (row per sensor)
    """
    cov_mtx = np.zeros((x.shape[0], x.shape[0]), dtype=np.complex)
    for j in range(x.shape[1]):
        cov_mtx += np.outer(x[:,j], x[:,j].conj())
    cov_mtx /= x.shape[1]
    return cov_mtx


def custom_delay_sum(in_data, fs=44100, n_search_directions=37, plot_bool=False):
    """
    Performs a delay-sum beamformer around the microphone array in the direction of:
    2 -- 1      315 - 0 - 45
    |    |      270       90
    3 -- 4      225- 180- 135
    :param in_data: input data, multiple channels supported.
    :param fs: sample rate, HZ
    :param plot_bool: boolean to control where plots are generated
    :param n_search_directions: int to dictate how many search directions to look
    :return:
    """
    print(">> Performing the Delay and Sum beamforming method")

    n_cols = np.shape(in_data)[1]
    n_rows = np.shape(in_data)[0]

    print("Shape of bandpass data: ", np.shape(in_data))

    delayed_array = np.zeros((n_rows, n_cols))

    # Instantiate variables
    c = 340     # m/s
    d = 0.045   # m
    h = np.sqrt(2 * d**2)
    pos = np.asarray([[0, 0], [-d, 0], [-d, -d], [0, -d]])  # Square Array, (1, 2, 3, 4)
    dt = 1 / fs
    time_across = d / c
    time_across_h = h / c
    # print("time: ", time_across)
    # print("samples across: ", time_across / dt)
    # print("hyp: ", h)
    # print("time: ", time_across_h)
    # print("samples across: ", time_across_h / dt)

    round_sample_h = round(time_across_h / dt)
    round_sample_s = round(time_across / dt)

    # print("round h ", round_sample_h)
    # print("round s ", round_sample_s)

    search_range = np.linspace(0, 2*np.pi, n_search_directions)[:-1]     # angles to check for incoming wave
    searched_array = np.zeros((n_rows, n_search_directions-1))

    # print("\n")
    # Code searches clockwise (sweeping from 1 to 4, to 3, to 2, back to 1)
    j = 0
    for theta in search_range:
        padded_chs = []
        # print("angle: ", theta * 180 / np.pi)
        for i in range(0, len(pos)-1):
            dist_x = pos[0][0] - pos[i+1][0]        # x Distance from reference mic to other mic
            dist_y = pos[0][1] - pos[i + 1][1]      # y Distance from reference mic to other mic

            delay_dist = dist_x * np.sin(theta) + dist_y * np.cos(theta)
            delay_time = delay_dist / c
            delay_sample = delay_time * fs

            # print("\n-----------")
            # print("Mic 1 and", i+2)
            # print("Theta: ", theta*180/np.pi)
            # print("sample delay: ", delay_sample)
            sample_remainder = delay_sample % 1
            if sample_remainder < 0.01:
                sample_remainder = 0


            # print("Distance between the mics: ", delay_dist)
            # print("time delay: ", delay_time)

            delay_sample = round(delay_sample)
            ch_current = in_data[:, i+1]
            ch_delay = shift(ch_current, -delay_sample, cval=0)
            delayed_array[:, i+1] = ch_delay
            # print("\n")

        delayed_array[:, 0] = in_data[:, 0]
        ch_norm = np.linalg.norm(np.sum(delayed_array, axis=1))  # add each channel together
        searched_array[:, j] = ch_norm

        j += 1

        if plot_bool:
            plt.plot(delayed_array)
            plt.legend()
            plt.show()

    # print(np.shape(searched_array))
    # f, pxx = welch_method(searched_array, fs, nperseg=len(searched_array)/4, noverlap=len(searched_array)/8,
    #                       print_info=False)

    # print(searched_array)
    max_values = np.amax(searched_array)
    # print(max)
    index = np.where(searched_array == max_values)
    direction = index[1][0]
    print("Direction of arrival: ", search_range[direction]*180/np.pi)
    doa = search_range[direction]*180/np.pi
    # plt.semilogx(f, pxx, label=["Ch1"])
    # plt.legend()
    # plt.show()

    return doa


def bf_delay_sum(in_data, sd, fs, shade, print_info=False):
    d = 0.045  # m
    pos = np.asarray([[0, 0, 0], [-d, 0, 0], [-d, -d, 0], [0, -d, 0]])  # Square Array, (1, 2, 3, 4)

    a2 = bf.steering_plane_wave(pos, 343, sd)
    y = bf.delay_and_sum(np.transpose(in_data), fs, a2)
    y = np.transpose(y)

    norms = np.linalg.norm(y, axis=0)
    max_value = np.amax(norms)
    index = np.where(norms == max_value)  # depending on freq resolution and sampling rate, len(index) may be >1
    n_directions = len(index[0])

    directions = []
    # for i in range(0, n_directions):
    #     direction = index[0][i]
    #     doa = sd[direction][0] * 180 / np.pi
    #     directions.append(doa)

    direction = index[0][0]
    doa = sd[direction][0] * 180 / np.pi
    directions.append(doa)
    mean_doa = np.mean(directions)
    print(">> Mean direction of arrival from " + str(n_directions) + " bin(s): " + str(mean_doa) + " degrees")
    return mean_doa


def bf_music(in_data, sd, fs, print_info=False):
    # https://github.com/vslobody/MUSIC/blob/master/music.py
    n_samples = in_data.shape[0]
    n_cols = in_data.shape[1]

    in_data = in_data
    print("Shape indata: ", in_data.shape)

    # things that will probably become inputs
    d = 0.045       # m
    m = 0         # idk where this comes from yet. number of sources?
    c = 343         # m/s
    f_start = 435  # Hz
    f_stop = 445    # Hz
    target_frequency = 440
    wavelength = c / target_frequency
    pl = d / wavelength

    pos = np.asarray([[0, 0, 0], [-pl, 0, 0], [-pl, -pl, 0], [0, -pl, 0]])  # Square Array, (1, 2, 3, 4)

    hermitian = np.transpose(np.conj(in_data))
    print("indata shape: ", in_data.shape)
    print("hermitian shape: ", hermitian.shape)
    rxx = in_data @ hermitian
    print("Rxx shape:", rxx.shape)

    e_values, e_vectors = la.eig(rxx)

    plt.plot(e_vectors)
    plt.show()

    print("e_values shape: ", e_values.shape)

    idx = e_values.argsort()[::-1]
    lam = e_values[idx]                             # Vector of sorted eigenvalues
    e_vectors = e_vectors[:, idx]                   # Sort eigenvectors accordingly
    e_vectors_n = e_vectors[:, m:len(e_vectors)]    # Noise eigenvectors (ASSUMPTION: M IS KNOWN)

    # MUSIC search directions
    azimuth_search = np.arange(0, 359, 5)           # Azimuth values to search
    elevation_search = [0]                          # placeholder, we do not do elevation

    # Wave-number vectors (in units of wavelength/2)
    x1 = np.cos(np.multiply(azimuth_search, np.pi / 180.))
    x2 = np.sin(np.multiply(azimuth_search, np.pi / 180.))
    x3 = np.sin(np.multiply(azimuth_search, 0.))
    x = [x1, x2, x3]
    k_search = np.multiply(x, 2 * np.pi / (c / ((f_stop + f_start) / 2)))

    ku = np.dot(pos, k_search)
    print("ku shape: ", ku)

    a_search = np.exp(np.multiply(ku, -1j))

    chemodan = np.dot(np.transpose(a_search), e_vectors_n)

    aac = np.absolute(chemodan)
    aad = np.square(aac)
    aae = np.sum(aad, 1)
    z = aae

    p = np.unravel_index(z.argmin(), z.shape)
    print("DOA: ", azimuth_search[p])
    # print("shape z: ", z.shape)
    plt.plot(azimuth_search, z)
    plt.show()

    return azimuth_search, z

    # a2 = bf.steering_plane_wave(pos, 343, sd)
    # y = bf.music(np.transpose(in_data), fs, a2)
    # y = np.transpose(y)
    #
    # max_value = np.amax(y)
    # print(max_value)
    # index = np.where(y == max_value)  # depending on freq resolution and sampling rate, len(index) may be >1
    # print(index)
    # n_directions = len(index[0])
    # print(n_directions)
    #
    # directions = []
    # for i in range(0, n_directions):
    #     direction = index[0][i]
    #     doa = sd[direction][0] * 180 / np.pi
    #     directions.append(doa)
    #
    # mean_doa = np.mean(directions)
    # print(">> Mean direction of arrival from " + str(n_directions) + " bin(s): " + str(mean_doa) + " degrees")
    # return mean_doa


def music2(in_data, fs):
    print("Performing music2")
    print("data shape: ", in_data.shape)
    d = 0.045
    x = np.asarray([0, -d, -d, 0])
    y = np.asarray([0, 0, -d, -d])
    incident_angles = np.arange(0, 360, 5)
    scanning_vectors = gen_scanning_vectors(4, x, y, incident_angles)
    print("scanning: ", scanning_vectors.shape)
    r = estimate_corr_matrix(in_data)
    adort = doa_music(r, scanning_vectors, 1)
    capon = doa_capon(r, scanning_vectors)
    # print("ADORT:", adort)
    # plt.plot(incident_angles, adort)
    plt.plot(incident_angles, capon)
    plt.show()


    # L = 1
    # N = 32  # number of ULA elements
    # array = np.linspace(0, (N - 1) / 2, N)
    #
    # Thetas = np.pi * (np.random.rand(L) - 1 / 2)  # random source directions
    # Alphas = np.random.randn(L) + np.random.randn(L) * 1j  # random source powers
    # Alphas = np.sqrt(1 / 2) * Alphas
    #
    # # CovMat is the signal covariance matrix, L is the number of sources, N is the number of antennas
    # # array holds the positions of antenna elements
    # # Angles are the grid of directions in the azimuth angular domain
    # _, V = la.eig(CovMat)
    # Qn = V[:, L:N]
    # numAngles = Angles.size
    # pspectrum = np.zeros(numAngles)
    # for i in range(numAngles):
    #     av = array_response_vector(array, Angles[i])
    #     pspectrum[i] = 1 / LA.norm((Qn.conj().transpose() @ av))
    # psindB = np.log10(10 * pspectrum / pspectrum.min())
    # DoAsMUSIC, _ = ss.find_peaks(psindB, height=1.35, distance=1.5)
    # return DoAsMUSIC, pspectrum


def custom_music(in_data):
    r = covariance(in_data)
    n_signals = 1

    if np.linalg.cond(r) > 10000:
        r += np.random.normal(0, np.max(np.abs(r)/1000000), r.shape)

    a, b = np.linalg.eigh(r)
    idx = a.argsort()[::-1]
    lam = a[idx]  # Sorted vector of eigenvalues
    b = b[:, idx]  # Eigenvectors rearranged accordingly
    en = b[:, n_signals:len(b)]  # Noise eigenvectors

    v = np.matmul(en, en.conj().T)
    return np.array([1.0/a[j].conj().dot(v).dot(a[j]).real for j in range(a.shape[0])])


def gcc_phat(sig, ref_sig, fs=1, max_tau=None, interp=16):
    '''
        This function computes the offset between the signal sig and the reference signal refsig
        using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''

    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + ref_sig.shape[0]

    # Generalized Cross Correlation Phase Transform
    gcc_sig = np.fft.rfft(sig, n=n)
    gcc_ref_sig = np.fft.rfft(ref_sig, n=n)
    r = gcc_sig * np.conj(gcc_ref_sig)

    cc = np.fft.irfft(r / np.abs(r), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift + 1]))

    # find max cross correlation index
    samp_shift = np.argmax(np.abs(cc)) - max_shift

    tau = samp_shift / float(interp * fs)

    return tau, cc
