from PyEMD import EMD, EEMD, CEEMDAN
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import csv
import operator
import pandas as pd
from scipy.stats import kurtosis, pearsonr
from scipy.signal import medfilt


def mmf(signal, alpha=0.2):
    return (1 - alpha) * np.median(signal) + alpha * np.mean(signal)


def mean_median_filter(signal, window=300, alpha=0.6):
    # Always odd  window
    if window % 2 == 0:
        window = window - 1

    # Appended signal
    new_signal = []
    for i in range((window - 1) / 2):
        new_signal.append(signal[0])
    for i in range(len(signal)):
        new_signal.append(signal[i])
    for i in range((window - 1) / 2):
        new_signal.append(signal[-1])

    # Windowing
    mmfoutput = []
    for i in range(len(signal)):
        mmfoutput.append(mmf(new_signal[i:i + window], alpha))

    return mmfoutput


def artificial_baseline(signal, f=4, noise_amplitude=40):
    signal = np.array(signal)
    Fs = 8000
    sample = len(signal)
    x = np.arange(sample)
    artificialbaseline = np.sin(2 * np.pi * x * f / Fs) + np.sin(2 * np.pi * x * f * 2 / Fs)
    # plt.plot(artificialbaseline)
    # plt.xlabel('Sample')
    # plt.ylabel('Amplitude')
    # plt.show()
    return noise_amplitude * artificialbaseline


def snr(signal, noise):
    S = np.std(signal)
    N = np.std(noise)
    SNR = 20 * np.log10(S / N)
    return SNR


def mean_median_filt(signal):
    baseline = mean_median_filter(signal, 250, 0.6)
    baseline = mean_median_filter(baseline, 600, 0.6)
    baseline = np.array(baseline)
    return baseline


def smothing_baseline(original_baseline_components, original_baseline):
    Q1 = 0
    Q2 = 0
    IMF_sum_1 = np.zeros(original_baseline_components.shape[1])
    for i in range(original_baseline_components.shape[0] - 1, -1, -1):
        # Target is to find Q1 or M
        IMF_sum_1 = IMF_sum_1 + original_baseline_components[i]
        if np.mean(original_baseline - IMF_sum_1) < 0:
            # print 'We have Q1'
            Q1 = i
            break
        # print i, np.mean(baseline-IMF_sum_1)

    IMF_sum_2 = np.zeros(original_baseline_components.shape[1])
    for i in range(0, original_baseline_components.shape[0]):
        # Target is to find Q2 or J
        IMF_sum_2 = IMF_sum_2 + original_baseline_components[i]
        if np.mean(IMF_sum_2) < 0:
            # print 'We have Q2'
            Q2 = i
            break
        # print i, np.mean(IMF_sum_2)

    print Q1, Q2

    IMF_sum = np.zeros(original_baseline_components.shape[1])
    for i in range(0, (Q1 + Q2) / 2):
        IMF_sum = IMF_sum + original_baseline_components[i]

    return IMF_sum


def PRMSD(original_signal, reconstructed_signal):
    num = np.sum(np.square(original_signal - reconstructed_signal))
    den = np.sum(np.square(original_signal))
    ans = np.sqrt(num / den)
    return ans * 100


def MAE(original_signal, reconstructed_signal):
    return np.max(np.abs(original_signal - reconstructed_signal))


def correlate(original_signal, reconstructed_signal):
    a = np.matmul(original_signal, reconstructed_signal)
    b1 = np.sqrt(np.sum(np.square(original_signal)))
    b2 = np.sqrt(np.sum(np.square(reconstructed_signal)))
    b = np.multiply(b1, b2)
    return np.divide(a, b)


def rmse(original_signal, reconstructed_signal):
    ans = np.sqrt(np.mean(np.square(original_signal - reconstructed_signal)))
    return ans


def VMD(signal, alpha, tau, K, DC, init, tol):
    '''
    Input and Parameters:
    ---------------------
        signal  - the time domain signal (1D) to be decomposed
        alpha   - the balancing parameter of the data-fidelity constraint
        tau     - time-step of the dual ascent ( pick 0 for noise-slack )
        K       - the number of modes to be recovered
        DC      - true if the first mode is put and kept at DC (0-freq)
        init    - 0 = all omegas start at 0
        1 = all omegas start uniformly distributed
        2 = all omegas initialized randomly
        tol     - tolerance of convergence criterion; typically around 1e-6

    Output:
    -------
        u       - the collection of decomposed modes
        u_hat   - spectra of the modes
        omega   - estimated mode center-frequencies
    '''

    # Data Preparations

    # Period and sampling frequency of input signal
    save_T = len(signal)
    fs = 1 / float(save_T)

    # extend the signal by mirroring
    T = save_T
    f_mirror = []
    for i in range(0, T / 2):
        f_mirror.append(signal[T / 2 - 1 - i])
    for i in range(T / 2, 3 * T / 2):
        f_mirror.append(signal[i - T / 2])
    for i in range(T - 1, T / 2 - 1, -1):
        f_mirror.append(signal[i])
    f = f_mirror

    # Time Domain 0 to T (of mirrored signal)
    T = len(f)
    t = []
    for i in range(1, T + 1):
        t.append(i / float(T))

    # Spectral Domain discretization
    freqs = []
    for i in range(len(t)):
        freqs.append(t[i] - 0.5 - 1 / float(T))

    # Maximum number of iterations (if not converged yet, then it won't anyway)
    N = 500

    # For future generalizations: individual alpha for each mode
    Alpha = alpha * np.ones(K, dtype=complex)

    # Construct and center f_hat
    f_hat = np.fft.fftshift((np.fft.fft(f)))
    f_hat_plus = f_hat
    for i in range(T / 2):
        f_hat_plus[i] = 0

    # matrix keeping track of every iterant // could be discarded for mem
    u_hat_plus = np.zeros((N, len(freqs), K), dtype=complex)

    # Initialization of omega_k
    omega_plus = np.zeros((N, K))
    if init == 1:
        for i in range(K):
            omega_plus[0, i] = (0.5 / K) * (i)
    elif init == 2:
        omega_plus[0, :] = np.sort(np.exp(np.log(fs) + (np.log(0.5) - np.log(fs)) * np.random.rand(K)))
    else:
        omega_plus[0, :] = 0

    # if DC mode imposed, set its omega to 0
    if DC:
        omega_plus[0, 0] = 0

    # start with empty dual variables
    lambda_hat = np.zeros((N, len(freqs)), dtype=complex)

    # other inits
    eps = np.spacing(1)
    # update step
    uDiff = tol + eps
    # loop counter
    n = 1
    # accumulator
    sum_uk = 0

    # Main loop for iterative updates

    # not converged and below iterations limit
    while (uDiff > tol and n < N):

        # update first mode accumulator
        k = 0
        sum_uk = u_hat_plus[n - 1, :, K - 1] + sum_uk - u_hat_plus[n - 1, :, 0]
        a = f_hat_plus - sum_uk - np.divide(lambda_hat[n - 1, :], 2)
        b = 1 + Alpha[k] * np.square(freqs - omega_plus[n - 1, k])

        # update spectrum of first mode through Wiener filter of residuals
        u_hat_plus[n, :, k] = np.divide(a, b)
        del a, b

        # update first omega if not held at 0
        if ~DC:
            a1 = freqs[T / 2:T]
            a1 = np.array(a1)
            a1 = a1.reshape(a1.shape[0], 1)
            a2 = np.square(abs(u_hat_plus[n, T / 2:T, k]))
            a2 = a2.reshape(a2.shape[0], 1)
            a2 = np.transpose(a2)
            a = np.matmul(a2, a1)
            b = sum(np.square(abs(u_hat_plus[n, T / 2:T, k])))
            omega_plus[n, k] = np.divide(a, b)
            del a1, a2, a, b

        # update of any other mode
        for k in range(1, K):
            # sum_uk = sum_uk.reshape(sum_uk.shape[0],1)
            # sum_uk = np.transpose(sum_uk)

            # accumulator
            sum_uk = u_hat_plus[n, :, k - 1] + sum_uk - u_hat_plus[n - 1, :, k]

            # mode spectrum
            a = f_hat_plus - sum_uk - lambda_hat[n - 1, :] / 2
            b = 1 + Alpha[k] * np.square(freqs - omega_plus[n - 1, k])
            u_hat_plus[n, :, k] = np.divide(a, b)
            del a, b

            # center frequencies
            a1 = freqs[T / 2:T]
            a1 = np.array(a1)
            a1 = a1.reshape(a1.shape[0], 1)
            a1 = np.transpose(a1)
            a2 = np.square(abs(u_hat_plus[n, T / 2:T, k]))
            a2 = a2.reshape(a2.shape[0], 1)
            a = np.matmul(a1, a2)
            b = sum(np.square(abs(u_hat_plus[n, T / 2:T, k])))
            omega_plus[n, k] = np.divide(a, b)
            del a1, a2, a, b

        # Dual ascent
        lambda_hat[n, :] = lambda_hat[n - 1, :] + tau * (np.sum(u_hat_plus[n, :, :], axis=1) - f_hat_plus)

        # loop counter
        n = n + 1

        # converged yet?
        uDiff = eps
        for i in range(K):
            val1 = u_hat_plus[n - 1, :, i] - u_hat_plus[n - 2, :, i]
            val1 = val1.reshape(val1.shape[0], 1)
            val2 = u_hat_plus[n - 1, :, i] - u_hat_plus[n - 2, :, i]
            val2 = val2.reshape(val2.shape[0], 1)
            val2 = np.transpose(np.conj(val2)) / T
            uDiff = uDiff + np.matmul(val2, val1)
            del val1, val2
        uDiff = abs(uDiff)

    # Postprocessing and cleanup

    # discard empty space if converged early
    N = min(N, n)
    omega = omega_plus[0:N, :]

    # Signal reconstruction
    u_hat = np.zeros((T, K), dtype=complex)
    u_hat[T / 2:T, :] = u_hat_plus[N - 1, T / 2:T, :]
    j = T / 2
    for i in range((T / 2), 1, -1):
        u_hat[i, :] = np.conj(u_hat_plus[N - 1, j, :])
        j = j + 1
    u_hat[0, :] = np.conj(u_hat[-1, :])
    u = np.zeros((K, len(t)))
    for k in range(K):
        u[k, :] = np.real(np.fft.ifft(np.fft.ifftshift(u_hat[:, k])))

    # remove mirror part
    u = u[:, T / 4:3 * T / 4]

    # recompute spectrum
    u_hat = np.zeros(np.transpose(u).shape, dtype=complex)
    for k in range(K):
        u_hat[:, k] = np.transpose(np.fft.fftshift(np.fft.fft(u[k, :])))

    return u, u_hat, omega
