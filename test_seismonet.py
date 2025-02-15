import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from obspy import read
import h5py
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
import joblib

# Define dataset paths
DATA_DIR = "seismonet_data"
STEAD_FILE = os.path.join(DATA_DIR, "stead_dataset.hdf5")
NCS_FILE = os.path.join(DATA_DIR, "ncs_seismic_data.mseed")
MODEL_PATH = "seismonet_trained.h5"

# Load trained model
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}. Please train it first.")
    return tf.keras.models.load_model(MODEL_PATH)

# Load trained NSWF wavelet
def load_trained_wavelet():
    return joblib.load("trained_wavelet_nswf.pkl")

# Load NCS India Dataset
def load_ncs_data(ncs_file_path, num_samples=1000):
    signals = []
    for i in range(num_samples):
        try:
            st = read(ncs_file_path)  # Read seismograph data
            tr = st[0]  # Extract first trace
            data = tr.data
            if len(data) > 4000:
                data = data[:4000]  # Truncate or pad to match 4000 samples
            elif len(data) < 4000:
                data = np.pad(data, (0, 4000 - len(data)), 'constant')
            signals.append(data)
        except:
            continue
    return np.array(signals).reshape((len(signals), 4000, 1))

# Load STEAD Dataset
def load_stead_data(stead_file_path, num_samples=1000):
    signals = []
    with h5py.File(stead_file_path, 'r') as f:
        waveforms = f["waveforms"]
        for i in tqdm(range(num_samples), desc="Loading STEAD Data"):
            try:
                data = waveforms[i]
                if len(data) > 4000:
                    data = data[:4000]  # Truncate or pad to match 4000 samples
                elif len(data) < 4000:
                    data = np.pad(data, (0, 4000 - len(data)), 'constant')
                signals.append(data)
            except:
                continue
    return np.array(signals).reshape((len(signals), 4000, 1))

# Evaluate model performance
def evaluate_model(model, dataset, dataset_name):
    print(f"\nEvaluating SeismoNet on {dataset_name} dataset...")
    predictions = model.predict(dataset)
    
    # Compute Metrics
    mse = mean_squared_error(dataset.flatten(), predictions.flatten())
    pcc, _ = pearsonr(dataset.flatten(), predictions.flatten())

    print(f"\nResults for {dataset_name}:")
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Pearson Correlation Coefficient (PCC): {pcc:.4f}")

    return predictions, mse, pcc

# Plot test results
def plot_results(original, noisy, denoised, dataset_name, sample_idx=0):
    plt.figure(figsize=(12, 6))

    # Original Seismic Signal
    plt.subplot(3, 1, 1)
    plt.plot(original[sample_idx], label="Original Signal", color='b')
    plt.legend()
    plt.title(f"{dataset_name} - Original Seismic Signal")

    # Noisy Signal
    plt.subplot(3, 1, 2)
    plt.plot(noisy[sample_idx], label="Noisy Signal", color='r', alpha=0.6)
    plt.legend()
    plt.title(f"{dataset_name} - Noisy Seismic Signal")

    # Denoised Signal (SeismoNet Output)
    plt.subplot(3, 1, 3)
    plt.plot(denoised[sample_idx], label="Denoised Signal", color='g')
    plt.legend()
    plt.title(f"{dataset_name} - SeismoNet Denoised Signal")

    plt.tight_layout()
    plt.show()

# Main function for testing
def test_seismonet():
    # Load trained model
    model = load_model()

    # Load test datasets
    ncs_test_data = load_ncs_data(NCS_FILE, num_samples=1000)
    stead_test_data = load_stead_data(STEAD_FILE, num_samples=1000)

    # Evaluate SeismoNet on NCS India dataset
    ncs_denoised, ncs_mse, ncs_pcc = evaluate_model(model, ncs_test_data, "NCS India")
    plot_results(ncs_test_data, ncs_test_data, ncs_denoised, "NCS India")

    # Evaluate SeismoNet on STEAD dataset
    stead_denoised, stead_mse, stead_pcc = evaluate_model(model, stead_test_data, "STEAD")
    plot_results(stead_test_data, stead_test_data, stead_denoised, "STEAD")

    print("\nEvaluation Completed.")

# Run Testing
if __name__ == "__main__":
    test_seismonet()
