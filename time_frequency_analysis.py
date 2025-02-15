import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pywt
import joblib
import tftb.processing as tfb
from sklearn.decomposition import PCA, NMF
import emd

# Load the trained wavelet from Custom_Wavelet_NSWF.py
def load_trained_wavelet():
    trained_wavelet = joblib.load("trained_wavelet_nswf.pkl")  # Ensure you have this file
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

# Apply Time-Frequency Analysis Methods
def compute_time_frequency_methods(t, x_n, fs, trained_wavelet):
    n_samples = len(x_n)
    
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

    # Smoothed Pseudo Wigner-Ville Distribution (SPWVD)
    spwvd = tfb.PseudoWignerVilleDistribution(x_n, smoothing=True)
    spwvd_matrix = spwvd.run()

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

    # Proposed NSWF (Using the trained wavelet)
    wavelet_coeffs = np.fft.fft(x_n) * np.fft.fft(trained_wavelet)
    x_nswf = np.real(np.fft.ifft(wavelet_coeffs))

    return (f_stft, t_stft, np.abs(Zxx)), (freqs, cwt_mat), (t_st, f_st, np.abs(st_matrix)), \
           (wvd_matrix, spwvd_matrix, cwd_matrix, rd_matrix), x_pca, x_emd, x_nmf, x_nswf

# Main Execution
fs = 100  # Sampling frequency
duration = 30  # Signal duration
snr_db = 30  # Noise level

t, s_n = generate_seismic_signal(fs, duration)
x_n = add_awgn(s_n, snr_db)

# Load the trained NSWF
trained_wavelet = load_trained_wavelet()

# Compute Time-Frequency Methods
stft_data, cwt_data, st_data, wvd_data, x_pca, x_emd, x_nmf, x_nswf = compute_time_frequency_methods(t, x_n, fs, trained_wavelet)

# Plot Results (12 subplots)
plt.figure(figsize=(14, 18))

# Original and Noisy Signal
plt.subplot(6, 2, 1)
plt.plot(t, s_n, label="Clean Signal", color='b')
plt.legend()
plt.title("Synthetic Seismic Signal s(n)")

plt.subplot(6, 2, 2)
plt.plot(t, x_n, label="Noisy Signal (s(n) + AWGN)", color='r', alpha=0.6)
plt.legend()
plt.title("Noisy Signal (30dB SNR)")

# STFT
plt.subplot(6, 2, 3)
plt.pcolormesh(stft_data[1], stft_data[0], stft_data[2], shading='auto')
plt.title("STFT")

# CWT
plt.subplot(6, 2, 4)
plt.pcolormesh(t, cwt_data[0], np.abs(cwt_data[1]), shading='auto')
plt.title("CWT")

# S-Transform
plt.subplot(6, 2, 5)
plt.pcolormesh(st_data[0], st_data[1], st_data[2], shading='auto')
plt.title("S-Transform")

# Wigner-Ville Distribution (WVD)
plt.subplot(6, 2, 6)
plt.imshow(wvd_data[0], aspect='auto', extent=[0, duration, 0, fs//2])
plt.title("WVD")

# Smoothed Pseudo WVD (SPWVD)
plt.subplot(6, 2, 7)
plt.imshow(wvd_data[1], aspect='auto', extent=[0, duration, 0, fs//2])
plt.title("SPWVD")

# Choi-Williams Distribution (CWD)
plt.subplot(6, 2, 8)
plt.imshow(wvd_data[2], aspect='auto', extent=[0, duration, 0, fs//2])
plt.title("CWD")

# Rihaczek Distribution (RD)
plt.subplot(6, 2, 9)
plt.imshow(wvd_data[3], aspect='auto', extent=[0, duration, 0, fs//2])
plt.title("Rihaczek Distribution (RD)")

# PCA
plt.subplot(6, 2, 10)
plt.plot(t, x_pca, label="PCA", color='g')
plt.legend()
plt.title("Principal Component Analysis (PCA)")

# EMD & NMF
plt.subplot(6, 2, 11)
plt.plot(t, x_emd, label="EMD", color='g')
plt.plot(t, x_nmf, label="NMF", color='orange')
plt.legend()
plt.title("EMD & NMF")

# Proposed NSWF
plt.subplot(6, 2, 12)
plt.plot(t, x_nswf, label="Proposed NSWF", color='g')
plt.legend()
plt.title("Proposed NSWF Time-Frequency Representation")

plt.tight_layout()
plt.show()
