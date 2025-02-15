import numpy as np
import matplotlib.pyplot as plt

def custom_wavelet(n, f0=10, delta_f=5, sigma=0.1, delta_sigma=0.05, beta=2, fs=100):
    """
    Generates a Non-Standard Wavelet Function (NSWF).
    
    Parameters:
        n (array): Time indices.
        f0 (float): Central frequency.
        delta_f (float): Frequency variation.
        sigma (float): Time localization parameter.
        delta_sigma (float): Adaptive parameter for time scaling.
        beta (float): Decay rate control.
        fs (int): Sampling frequency.

    Returns:
        wavelet (complex array): Computed wavelet values.
    """
    Ts = 1 / fs
    t = n * Ts
    wavelet = np.exp(1j * 2 * np.pi * (f0 + delta_f) * t) * \
              np.exp(-((t / (sigma + delta_sigma)) ** (2 * beta)))
    return wavelet

# Visualization
n = np.arange(-50, 50, 1)
wavelet = custom_wavelet(n)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(n, np.real(wavelet), label="Real Part")
plt.plot(n, np.imag(wavelet), label="Imaginary Part")
plt.legend()
plt.title("Custom Wavelet Function (NSWF)")

plt.subplot(1, 2, 2)
plt.specgram(np.real(wavelet), Fs=100, cmap='inferno')
plt.title("Spectrogram of NSWF")
plt.show()
