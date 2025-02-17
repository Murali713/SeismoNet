import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt
import tftb.processing as tfb
from sklearn.decomposition import PCA, NMF
import emd
from scipy.ndimage import median_filter
from scipy.signal import wiener, convolve
from Custom_Wavelet_NSWF import custom_wavelet  # Import NSWF function

# Step 1: Generate Seismic Signal (Using Ricker Wavelet)
def generate_seismic_signal(fs=100, duration=30, fn=4, zeta=0.3):
    """
    Generates a synthetic seismic ground motion signal using a modulated Ricker wavelet.
    """
    t = np.linspace(0, duration, fs * duration)
    ricker_wavelet = signal.ricker(len(t), fn * fs / (2 * np.sqrt(np.pi * zeta)))

    # Envelope function for smooth tapering
    envelope = np.exp(-((t - 0.4 * duration) ** 2) / (2 * (0.3 * duration) ** 2))

    s_n = 0.3 * ricker_wavelet * envelope  # Scale with standard deviation
    return t, s_n

# Step 2: Add White Gaussian Noise (AWGN)
def add_awgn(signal, snr_db):
    noise_power = np.var(signal) / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), signal.shape)
    return signal + noise

# Step 3: Apply NSWF-Based Denoising using Convolution
def apply_nswf_denoising(signal, fs):
    """
    Uses the Non-Standard Wavelet Function (NSWF) to denoise the signal.
    """
    n = np.arange(-len(signal) // 2, len(signal) // 2)
    trained_wavelet = custom_wavelet(n, fs=fs)  # Generate NSWF dynamically
    trained_wavelet = np.real(trained_wavelet)  # Use real part for filtering

    # Perform convolution-based wavelet denoising
    denoised_signal = convolve(signal, trained_wavelet, mode="same") / np.sum(trained_wavelet)
    return denoised_signal

# Step 4: Compute Time-Frequency Methods
def compute_time_frequency_methods(t, x_n, fs):
    """
    Computes various time-frequency methods for seismic signal analysis.
    """
    # Short-Time Fourier Transform (STFT)
    f_stft, t_stft, Zxx = signal.stft(x_n, fs=fs, nperseg=256)

    # Continuous Wavelet Transform (CWT)
    scales = np.arange(1, 128)
    cwt_mat, freqs = pywt.cwt(x_n, scales, 'morl', sampling_period=1/fs)

    # S-Transform
    st = tfb.STFT(x_n)
    t_st, f_st, st_matrix = st.run()

    # Wigner-Ville Distribution (WVD)
    wvd = tfb.WignerVilleDistribution(x_n)
    wvd_matrix = wvd.run()

    # Choi-Williams Distribution (CWD)
    cwd = tfb.CohenChoiWilliamsDistribution(x_n)
    cwd_matrix = cwd.run()

    # Rihaczek Distribution (RD)
    rd = tfb.RihaczekDistribution(x_n)
    rd_matrix = rd.run()

    # Principal Component Analysis (PCA)
    pca = PCA(n_components=1)
    x_pca = pca.fit_transform(x_n.reshape(-1, 1)).flatten()

    # Empirical Mode Decomposition (EMD)
    imfs = emd.sift.sift(x_n)
    x_emd = np.sum(imfs, axis=0)

    # Non-Negative Matrix Factorization (NMF)
    nmf = NMF(n_components=1, init='random', random_state=0)
    x_nmf = nmf.fit_transform(x_n.reshape(-1, 1)).flatten()

    # Proposed NSWF Filtering
    x_nswf = apply_nswf_denoising(x_n, fs)

    return (f_stft, t_stft, np.abs(Zxx)), (freqs, cwt_mat), (t_st, f_st, np.abs(st_matrix)), \
           (wvd_matrix, cwd_matrix, rd_matrix), x_pca, x_emd, x_nmf, x_nswf

# Step 5: Main Execution
fs = 100  # Sampling frequency
duration = 30  # Duration of the earthquake signal
snr_db = 30  # Noise level in dB

t, s_n = generate_seismic_signal(fs, duration)  # Generate seismic signal
x_n = add_awgn(s_n, snr_db)  # Add noise

# Compute Time-Frequency Methods
stft_data, cwt_data, st_data, wvd_data, x_pca, x_emd, x_nmf, x_nswf = compute_time_frequency_methods(t, x_n, fs)

# Step 6: Plot Results (12 subplots)
plt.figure(figsize=(14, 20))

# 1. Clean Seismic Signal
plt.subplot(6, 2, 1)
plt.plot(t, s_n, label="Clean Seismic Signal", color='b')
plt.legend()
plt.title("Synthetic Seismic Signal")

# 2. Noisy Signal
plt.subplot(6, 2, 2)
plt.plot(t, x_n, label="Noisy Signal (s(n) + AWGN)", color='r', alpha=0.6)
plt.legend()
plt.title("Noisy Seismic Signal (30dB SNR)")

# 3. STFT
plt.subplot(6, 2, 3)
plt.pcolormesh(stft_data[1], stft_data[0], stft_data[2], shading='auto')
plt.title("Short-Time Fourier Transform (STFT)")

# 4. Continuous Wavelet Transform (CWT)
plt.subplot(6, 2, 4)
plt.pcolormesh(t, cwt_data[0], np.abs(cwt_data[1]), shading='auto')
plt.title("Continuous Wavelet Transform (CWT)")

# 5. S-Transform
plt.subplot(6, 2, 5)
plt.pcolormesh(st_data[0], st_data[1], st_data[2], shading='auto')
plt.title("S-Transform")

# 6. Wigner-Ville Distribution (WVD)
plt.subplot(6, 2, 6)
plt.imshow(wvd_data[0], aspect='auto', extent=[0, duration, 0, fs//2])
plt.title("Wigner-Ville Distribution (WVD)")

# 7. Choi-Williams Distribution (CWD)
plt.subplot(6, 2, 7)
plt.imshow(wvd_data[1], aspect='auto', extent=[0, duration, 0, fs//2])
plt.title("Choi-Williams Distribution (CWD)")

# 8. Rihaczek Distribution (RD)
plt.subplot(6, 2, 8)
plt.imshow(wvd_data[2], aspect='auto', extent=[0, duration, 0, fs//2])
plt.title("Rihaczek Distribution (RD)")

# 9. Principal Component Analysis (PCA)
plt.subplot(6, 2, 9)
plt.plot(t, x_pca, label="PCA", color='g')
plt.legend()
plt.title("Principal Component Analysis (PCA)")

# 10. Empirical Mode Decomposition (EMD)
plt.subplot(6, 2, 10)
plt.plot(t, x_emd, label="EMD", color='g')
plt.legend()
plt.title("Empirical Mode Decomposition (EMD)")

# 11. Non-Negative Matrix Factorization (NMF)
plt.subplot(6, 2, 11)
plt.plot(t, x_nmf, label="NMF", color='orange')
plt.legend()
plt.title("Non-Negative Matrix Factorization (NMF)")

# 12. Proposed NSWF Filtering
plt.subplot(6, 2, 12)
plt.plot(t, x_nswf, label="Proposed NSWF", color='g')
plt.legend()
plt.title("Proposed NSWF Filtering")

plt.tight_layout()
plt.show()
