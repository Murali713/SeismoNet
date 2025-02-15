import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt
import joblib
from scipy.ndimage import median_filter
from scipy.signal import wiener, convolve

# Load the trained wavelet from Custom_Wavelet_NSWF.py
def load_trained_wavelet():
    """
    Loads the trained wavelet coefficients from the saved NSWF model.
    Returns:
        trained_wavelet (array): Trained Non-Standard Wavelet Function.
    """
    trained_wavelet = joblib.load("trained_wavelet_nswf.pkl")  # Ensure this file is available
    return trained_wavelet

# Generate Synthetic Seismic Signal using Ricker Wavelet
def generate_seismic_signal(fs=100, duration=30, f0=10):
    t = np.linspace(0, duration, int(fs * duration))
    s_n = (1 - 2 * (np.pi * f0 * t)**2) * np.exp(- (np.pi * f0 * t)**2)
    return t, s_n

# Add White Gaussian Noise (AWGN)
def add_awgn(signal, snr_db):
    noise_power = np.var(signal) / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

# Apply NSWF-based Denoising using the trained wavelet
def apply_nswf_denoising(signal, trained_wavelet):
    """
    Uses the trained Non-Standard Wavelet Function (NSWF) to denoise the signal.

    Parameters:
        signal (array): Noisy input signal.
        trained_wavelet (array): Learned wavelet from SeismoNet.

    Returns:
        denoised_signal (array): Filtered signal.
    """
    wavelet_coeffs = np.fft.fft(signal) * np.fft.fft(trained_wavelet)
    threshold = np.mean(np.abs(wavelet_coeffs)) + 0.5 * np.std(wavelet_coeffs)
    wavelet_coeffs[np.abs(wavelet_coeffs) < threshold] = 0
    return np.real(np.fft.ifft(wavelet_coeffs))

# Apply Different Denoising Methods
def denoise_signal(x_n, trained_wavelet):
    # Wiener Filtering
    x_wiener = wiener(x_n, mysize=5)
    
    # Median Filtering
    x_median = median_filter(x_n, size=5)

    # Soft Thresholding using Wavelet Transform
    coeffs = pywt.wavedec(x_n, 'db4', level=4)
    coeffs_thresh = [pywt.threshold(c, value=np.std(c), mode='soft') for c in coeffs]
    x_soft = pywt.waverec(coeffs_thresh, 'db4')

    # Hard Thresholding
    coeffs_thresh_hard = [pywt.threshold(c, value=np.std(c), mode='hard') for c in coeffs]
    x_hard = pywt.waverec(coeffs_thresh_hard, 'db4')

    # Gaussian Smoothing
    x_gaussian = signal.gaussian(len(x_n), std=5)
    x_gaussian = convolve(x_n, x_gaussian, mode='same') / np.sum(x_gaussian)

    # Proposed NSWF Filtering using Trained Wavelet
    x_nswf = apply_nswf_denoising(x_n, trained_wavelet)

    return x_wiener, x_median, x_soft, x_hard, x_gaussian, x_nswf

# Main Execution
fs = 100  # Sampling frequency
duration = 30  # Signal duration
snr_db = 30  # Noise level

t, s_n = generate_seismic_signal(fs, duration)
x_n = add_awgn(s_n, snr_db)

# Load the trained NSWF
trained_wavelet = load_trained_wavelet()

# Apply Denoising Methods
x_wiener, x_median, x_soft, x_hard, x_gaussian, x_nswf = denoise_signal(x_n, trained_wavelet)

# Compute Wavelet Coefficient Differences
coeff_diff_wiener = np.abs(x_n - x_wiener)
coeff_diff_median = np.abs(x_n - x_median)
coeff_diff_soft = np.abs(x_n - x_soft)
coeff_diff_hard = np.abs(x_n - x_hard)
coeff_diff_gaussian = np.abs(x_n - x_gaussian)
coeff_diff_nswf = np.abs(x_n - x_nswf)

# Plot Results (12 subplots)
plt.figure(figsize=(14, 14))

# Original and Noisy Signal
plt.subplot(6, 2, 1)
plt.plot(t, s_n, label="Clean Signal", color='b')
plt.legend()
plt.title("Synthetic Seismic Signal s(n)")

plt.subplot(6, 2, 2)
plt.plot(t, x_n, label="Noisy Signal (s(n) + AWGN)", color='r', alpha=0.6)
plt.legend()
plt.title("Noisy Signal (30dB SNR)")

# Wiener Filter
plt.subplot(6, 2, 3)
plt.plot(t, x_wiener, label="Wiener Filter", color='g')
plt.legend()
plt.title("Wiener Denoised Signal")

plt.subplot(6, 2, 4)
plt.plot(t, coeff_diff_wiener, label="Wiener Coefficient Difference", color='k')
plt.legend()
plt.title("Wiener Filter - Coefficient Difference")

# Median Filter
plt.subplot(6, 2, 5)
plt.plot(t, x_median, label="Median Filter", color='g')
plt.legend()
plt.title("Median Denoised Signal")

plt.subplot(6, 2, 6)
plt.plot(t, coeff_diff_median, label="Median Coefficient Difference", color='k')
plt.legend()
plt.title("Median Filter - Coefficient Difference")

# Soft Thresholding
plt.subplot(6, 2, 7)
plt.plot(t, x_soft, label="Soft Thresholding", color='g')
plt.legend()
plt.title("Soft Thresholding Denoised Signal")

plt.subplot(6, 2, 8)
plt.plot(t, coeff_diff_soft, label="Soft Threshold Coefficient Difference", color='k')
plt.legend()
plt.title("Soft Thresholding - Coefficient Difference")

# Hard Thresholding
plt.subplot(6, 2, 9)
plt.plot(t, x_hard, label="Hard Thresholding", color='g')
plt.legend()
plt.title("Hard Thresholding Denoised Signal")

plt.subplot(6, 2, 10)
plt.plot(t, coeff_diff_hard, label="Hard Threshold Coefficient Difference", color='k')
plt.legend()
plt.title("Hard Thresholding - Coefficient Difference")

# NSWF Filtering
plt.subplot(6, 2, 11)
plt.plot(t, x_nswf, label="NSWF Denoised Signal", color='g')
plt.legend()
plt.title("Proposed NSWF Denoised Signal")

plt.subplot(6, 2, 12)
plt.plot(t, coeff_diff_nswf, label="NSWF Coefficient Difference", color='k')
plt.legend()
plt.title("Proposed NSWF - Coefficient Difference")

plt.tight_layout()
plt.show()
