import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt
from scipy.ndimage import median_filter
from scipy.signal import wiener, convolve
from Custom_Wavelet_NSWF import custom_wavelet  # Import your NSWF function

# Step 1: Generate Seismic Signal (Using Ricker Wavelet)
def generate_seismic_signal(fs=100, duration=30, fn=4, zeta=0.3):
    """
    Generates a synthetic seismic ground motion signal using a modulated Ricker wavelet.

    Parameters:
        fs (int): Sampling frequency.
        duration (int): Duration of the earthquake motion (seconds).
        fn (float): Dominant frequency of the earthquake excitation (Hz).
        zeta (float): Bandwidth parameter.

    Returns:
        t (array): Time vector.
        s_n (array): Seismic ground motion signal.
    """
    t = np.linspace(0, duration, fs * duration)
    ricker_wavelet = signal.ricker(len(t), fn * fs / (2 * np.sqrt(np.pi * zeta)))  # Ricker wavelet (approximation)
    
    # Envelope function similar to MATLAB's 'seismSim' characteristics
    envelope = np.exp(-((t - 0.4 * duration) ** 2) / (2 * (0.3 * duration) ** 2))

    s_n = 0.3 * ricker_wavelet * envelope  # Scale with standard deviation
    return t, s_n

# Step 2: Add White Gaussian Noise (AWGN)
def add_awgn(signal, snr_db):
    """
    Adds White Gaussian Noise (AWGN) to a signal.

    Parameters:
        signal (array): Input clean signal.
        snr_db (float): Signal-to-noise ratio (SNR) in dB.

    Returns:
        noisy_signal (array): Noisy signal with AWGN.
    """
    noise_power = np.var(signal) / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

# Step 3: Apply NSWF-Based Denoising using Convolution
def apply_nswf_denoising(signal, fs):
    """
    Uses the Non-Standard Wavelet Function (NSWF) to denoise the signal.

    Parameters:
        signal (array): Noisy input signal.
        fs (int): Sampling frequency.

    Returns:
        denoised_signal (array): Filtered signal.
    """
    n = np.arange(-len(signal) // 2, len(signal) // 2)
    trained_wavelet = custom_wavelet(n, fs=fs)  # Generate NSWF dynamically
    trained_wavelet = np.real(trained_wavelet)  # Use real part for filtering

    # Perform convolution-based wavelet denoising
    denoised_signal = convolve(signal, trained_wavelet, mode="same") / np.sum(trained_wavelet)
    return denoised_signal

# Step 4: Apply Different Denoising Methods
def denoise_signal(x_n, fs):
    """
    Applies various denoising methods including NSWF.

    Parameters:
        x_n (array): Noisy input signal.
        fs (int): Sampling frequency.

    Returns:
        tuple: Denoised signals from different methods.
    """
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

    # Proposed NSWF Filtering
    x_nswf = apply_nswf_denoising(x_n, fs)

    return x_wiener, x_median, x_soft, x_hard, x_gaussian, x_nswf

# Step 5: Main Execution
fs = 100  # Sampling frequency
duration = 30  # Duration of the earthquake signal
snr_db = 30  # Noise level in dB

t, s_n = generate_seismic_signal(fs, duration)  # Generate seismic signal
x_n = add_awgn(s_n, snr_db)  # Add noise

# Apply Denoising Methods
x_wiener, x_median, x_soft, x_hard, x_gaussian, x_nswf = denoise_signal(x_n, fs)

# Compute Wavelet Coefficient Differences
coeff_diff_wiener = np.abs(x_n - x_wiener)
coeff_diff_median = np.abs(x_n - x_median)
coeff_diff_soft = np.abs(x_n - x_soft)
coeff_diff_hard = np.abs(x_n - x_hard)
coeff_diff_gaussian = np.abs(x_n - x_gaussian)
coeff_diff_nswf = np.abs(x_n - x_nswf)

# Step 6: Plot Results (12 subplots)
plt.figure(figsize=(14, 14))

# Original and Noisy Signal
plt.subplot(6, 2, 1)
plt.plot(t, s_n, label="Clean Seismic Signal", color='b')
plt.legend()
plt.title("Earthquake Ground Motion Simulation")

plt.subplot(6, 2, 2)
plt.plot(t, x_n, label="Noisy Signal (s(n) + AWGN)", color='r', alpha=0.6)
plt.legend()
plt.title("Noisy Seismic Signal (30dB SNR)")

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
