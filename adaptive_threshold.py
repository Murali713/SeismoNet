import numpy as np
import scipy.signal as signal
import pywt
from scipy.signal import convolve
from bidcg_optimizer import BiDCG  # Custom BiDCG Optimizer
from Custom_Wavelet_NSWF import custom_wavelet  # Import NSWF function

# Step 1: Compute Wavelet Coefficients
def compute_wavelet_coefficients(x, n, fs, delta_f, delta_sigma, beta):
    """
    Compute wavelet coefficients using the adaptive NSWF.
    """
    Ts = 1 / fs
    wavelet_conjugate = np.exp(-1j * 2 * np.pi * (delta_f) * (n - n[:, None]) * Ts) * \
                        np.exp(-((n - n[:, None]) * Ts / (delta_sigma)) ** (2 * beta))
    
    # Compute wavelet transform
    c = np.sum(x[:, None] * wavelet_conjugate, axis=0)
    
    return c

# Step 2: Compute Local Variance
def compute_local_variance(c, window_size=5):
    """
    Compute local variance of wavelet coefficients.
    """
    N = len(c)
    local_mean = np.convolve(c, np.ones(window_size) / window_size, mode='same')
    local_variance = np.convolve((c - local_mean) ** 2, np.ones(window_size) / window_size, mode='same')
    
    return local_variance

# Step 3: Adaptive Thresholding
def adaptive_threshold(c, local_variance, k=1.5):
    """
    Compute adaptive thresholding based on local variance.
    """
    mu = np.mean(c)
    threshold = mu + k * np.sqrt(local_variance)
    
    # Thresholding function
    c_thresholded = np.sign(c) * (np.abs(c) - threshold * (1 - (np.abs(c) ** 2 / local_variance).clip(0, 1)))
    return c_thresholded

# Step 4: Reconstruction from Thresholded Coefficients
def reconstruct_signal(c_thresholded, n, fs, delta_f, delta_sigma, beta):
    """
    Reconstruct signal using inverse wavelet transform.
    """
    Ts = 1 / fs
    wavelet = np.exp(1j * 2 * np.pi * (delta_f) * (n - n[:, None]) * Ts) * \
              np.exp(-((n - n[:, None]) * Ts / (delta_sigma)) ** (2 * beta))

    x_reconstructed = np.sum(c_thresholded[:, None] * wavelet, axis=0)
    
    return np.real(x_reconstructed)

# Step 5: Compute Loss Function
def compute_loss(x, x_reconstructed, c, local_variance, k=1.5):
    """
    Compute loss function for adaptive wavelet denoising.
    """
    N = len(x)
    mu = np.mean(c)
    loss = (1 / N) * np.sum((c - (mu + k * np.sqrt(local_variance)) - x) ** 2)
    
    return loss

# Step 6: Optimize Using BiDCG
def optimize_denoising(x, fs=100, delta_f=5, delta_sigma=0.1, beta=2, max_iter=50):
    """
    Apply BiDCG optimization for denoising process.
    """
    N = len(x)
    n = np.arange(N)

    # Compute Initial Wavelet Coefficients
    c = compute_wavelet_coefficients(x, n, fs, delta_f, delta_sigma, beta)

    # Compute Local Variance
    local_variance = compute_local_variance(c)

    # Adaptive Thresholding
    c_thresholded = adaptive_threshold(c, local_variance)

    # Signal Reconstruction
    x_reconstructed = reconstruct_signal(c_thresholded, n, fs, delta_f, delta_sigma, beta)

    # Compute Initial Loss
    loss = compute_loss(x, x_reconstructed, c, local_variance)

    # Apply BiDCG Optimization
    bidcg_optimizer = BiDCG(learning_rate=0.001)
    for _ in range(max_iter):
        grad = 2 * (c - (np.mean(c) + 1.5 * np.sqrt(local_variance)) - x)
        c -= bidcg_optimizer.apply_update(grad)

        # Recompute Signal and Loss
        x_reconstructed = reconstruct_signal(c, n, fs, delta_f, delta_sigma, beta)
        loss = compute_loss(x, x_reconstructed, c, local_variance)

    return x_reconstructed, loss

# Example Usage
if __name__ == "__main__":
    # Generate Example Seismic Signal
    fs = 100  # Sampling frequency
    duration = 30  # Duration in seconds
    t = np.linspace(0, duration, fs * duration)
    x = np.sin(2 * np.pi * 4 * t) + 0.3 * np.random.randn(len(t))  # Simulated seismic signal with noise

    # Apply SeismoNet TFR Denoising
    x_denoised, final_loss = optimize_denoising(x, fs=fs)

    # Plot Results
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, x, label="Noisy Seismic Signal", color='r')
    plt.legend()
    plt.title("Noisy Input Signal")

    plt.subplot(2, 1, 2)
    plt.plot(t, x_denoised, label="Denoised Signal (SeismoNet + NSWF)", color='g')
    plt.legend()
    plt.title("Denoised Seismic Signal")

    plt.tight_layout()
    plt.show()

    print(f"Final Loss after BiDCG Optimization: {final_loss:.6f}")
