import numpy as np
import matplotlib.pyplot as plt

# Step 1: Compute Wavelet Coefficients
def custom_wavelet(n, f0=10, delta_f=5, sigma=0.1, delta_sigma=0.05, beta=2, fs=100):
    """
    Computes the Non-Standard Wavelet Function (NSWF).

    Parameters:
        n (array): Time indices.
        f0 (float): Central frequency.
        delta_f (float): Frequency variation.
        sigma (float): Time localization parameter.
        delta_sigma (float): Adaptive time-scaling.
        beta (float): Decay rate control.
        fs (int): Sampling frequency.

    Returns:
        wavelet (complex array): Computed wavelet values.
    """
    Ts = 1 / fs
    t = n * Ts
    wavelet = np.exp(-1j * 2 * np.pi * (f0 + delta_f) * t) * \
              np.exp(-((t / (sigma + delta_sigma)) ** (2 * beta)))
    return wavelet

def compute_wavelet_coefficients(signal, wavelet):
    """
    Computes wavelet coefficients by convolving the input signal with the wavelet function.

    Parameters:
        signal (array): Input seismic signal.
        wavelet (array): Complex wavelet function.

    Returns:
        coefficients (array): Computed wavelet coefficients.
    """
    N = len(signal)
    coefficients = np.zeros(N, dtype=complex)
    for n in range(N):
        coefficients[n] = np.sum(signal * np.conjugate(np.roll(wavelet, n)))
    return coefficients

# Step 2: Calculate Local Variance
def compute_local_variance(coefficients):
    """
    Computes the local variance of wavelet coefficients.

    Parameters:
        coefficients (array): Wavelet coefficients.

    Returns:
        local_variance (float): Computed local variance.
    """
    mean_val = np.mean(coefficients)
    local_variance = np.sum((coefficients - mean_val) ** 2) / len(coefficients)
    return local_variance, mean_val

# Step 3: Adaptive Thresholding
def adaptive_threshold(coefficients, k=1.5):
    """
    Applies adaptive thresholding based on local variance.

    Parameters:
        coefficients (array): Wavelet coefficients.
        k (float): Scaling factor for threshold adjustment.

    Returns:
        thresholded_coeffs (array): Coefficients after thresholding.
    """
    sigma_local, mu = compute_local_variance(coefficients)
    threshold = mu + k * sigma_local
    thresholded_coeffs = np.where(np.abs(coefficients) > threshold, coefficients, 0)
    return thresholded_coeffs

# Step 4: Reconstruction using Inverse Wavelet Transform
def reconstruct_signal(thresholded_coeffs, wavelet):
    """
    Reconstructs the denoised signal using inverse wavelet transform.

    Parameters:
        thresholded_coeffs (array): Thresholded wavelet coefficients.
        wavelet (array): Wavelet function used for reconstruction.

    Returns:
        reconstructed_signal (array): Denoised seismic signal.
    """
    N = len(thresholded_coeffs)
    reconstructed_signal = np.zeros(N)
    for m in range(N):
        reconstructed_signal[m] = np.sum(thresholded_coeffs * wavelet)
    return np.real(reconstructed_signal)

# Step 5: Loss Function (Mean Squared Error)
def compute_loss(original_signal, denoised_signal):
    """
    Computes the Mean Squared Error (MSE) loss.

    Parameters:
        original_signal (array): Original seismic signal.
        denoised_signal (array): Reconstructed (denoised) signal.

    Returns:
        loss (float): Mean Squared Error value.
    """
    loss = np.mean((denoised_signal - original_signal) ** 2)
    return loss

# -------------------------- TEST THE IMPLEMENTATION --------------------------
# Generate a synthetic seismic signal
fs = 100  # Sampling frequency
N = 1024  # Number of samples
t = np.linspace(0, 10, N)
synthetic_signal = np.sin(2 * np.pi * 5 * t) + np.random.normal(0, 0.2, N)  # Noisy sine wave

# Generate wavelet function
wavelet = custom_wavelet(np.arange(N))

# Step 1: Compute wavelet coefficients
wavelet_coeffs = compute_wavelet_coefficients(synthetic_signal, wavelet)

# Step 2: Calculate local variance
local_variance, mean_coeff = compute_local_variance(wavelet_coeffs)

# Step 3: Apply adaptive thresholding
thresholded_coeffs = adaptive_threshold(wavelet_coeffs, k=1.5)

# Step 4: Reconstruct the denoised signal
denoised_signal = reconstruct_signal(thresholded_coeffs, wavelet)

# Step 5: Compute loss
mse_loss = compute_loss(synthetic_signal, denoised_signal)

# -------------------------- VISUALIZATION --------------------------
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(t, synthetic_signal, label="Noisy Seismic Signal", color='r')
plt.legend()
plt.title("Noisy Seismic Signal")

plt.subplot(3, 1, 2)
plt.plot(np.abs(wavelet_coeffs), label="Wavelet Coefficients", color='b')
plt.axhline(y=mean_coeff + 1.5 * local_variance, color='k', linestyle='dashed', label="Threshold")
plt.legend()
plt.title("Wavelet Coefficients & Thresholding")

plt.subplot(3, 1, 3)
plt.plot(t, denoised_signal, label="Denoised Signal", color='g')
plt.legend()
plt.title("Reconstructed Seismic Signal")

plt.tight_layout()
plt.show()

print(f"Mean Squared Error (MSE): {mse_loss:.6f}")
