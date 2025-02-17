import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tqdm import tqdm
from obspy import read
import h5py
from bidcg_optimizer import BiDCG  # Custom optimizer
from seismonet_decoder import build_decoder  # Decoder function
from seismonet_encoder import build_encoder  # Encoder function
from Custom_Wavelet_NSWF import custom_wavelet  # Import NSWF function
from scipy.signal import convolve

# Define dataset paths
DATA_DIR = "seismonet_data"
STEAD_FILE = os.path.join(DATA_DIR, "stead_dataset.hdf5")
NCS_FILE = os.path.join(DATA_DIR, "ncs_seismic_data.mseed")
MODEL_PATH = "seismonet_trained.h5"

# Ensure dataset directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Function to download STEAD dataset manually
def download_stead():
    print("Please download the STEAD dataset manually from Kaggle.")
    print(f"Once downloaded, place the file inside {DATA_DIR} with the name 'stead_dataset.hdf5'.")

# Function to request NCS dataset manually
def download_ncs():
    print("Please request access to the NCS dataset from the official NCS portal.")
    print(f"Once obtained, place the file inside {DATA_DIR} with the name 'ncs_seismic_data.mseed'.")

# Load NCS India Dataset
def load_ncs_data(ncs_file_path, num_samples=5000):
    signals = []
    for _ in range(num_samples):
        try:
            st = read(ncs_file_path)  # Read seismograph data
            tr = st[0]  # Extract first trace
            data = tr.data.astype(np.float32)

            # Ensure consistent signal length
            if len(data) > 4000:
                data = data[:4000]
            elif len(data) < 4000:
                data = np.pad(data, (0, 4000 - len(data)), 'constant')

            # Normalize the signal
            data = (data - np.mean(data)) / np.std(data)
            signals.append(data)
        except:
            continue
    return np.array(signals).reshape((len(signals), 4000, 1))

# Load STEAD Dataset
def load_stead_data(stead_file_path, num_samples=5000):
    signals = []
    with h5py.File(stead_file_path, 'r') as f:
        waveforms = f["waveforms"]
        for i in tqdm(range(num_samples), desc="Loading STEAD Data"):
            try:
                data = waveforms[i].astype(np.float32)

                # Ensure consistent signal length
                if len(data) > 4000:
                    data = data[:4000]
                elif len(data) < 4000:
                    data = np.pad(data, (0, 4000 - len(data)), 'constant')

                # Normalize the signal
                data = (data - np.mean(data)) / np.std(data)
                signals.append(data)
            except:
                continue
    return np.array(signals).reshape((len(signals), 4000, 1))

# Apply NSWF-Based Denoising using Convolution
def apply_nswf_denoising(signal, fs=100):
    """
    Uses the Non-Standard Wavelet Function (NSWF) to denoise the signal.
    """
    n = np.arange(-len(signal) // 2, len(signal) // 2)
    trained_wavelet = custom_wavelet(n, fs=fs)  # Generate NSWF dynamically
    trained_wavelet = np.real(trained_wavelet)  # Use real part for filtering

    # Perform convolution-based wavelet denoising
    denoised_signal = convolve(signal, trained_wavelet, mode="same") / np.sum(trained_wavelet)
    return denoised_signal

# Build Full SeismoNet Model (Encoder-Decoder)
def build_seismonet():
    input_signal = Input(shape=(4000, 1))

    # Encoder from seismonet_encoder.py
    encoded = build_encoder(input_signal)

    # Bottleneck
    flatten = Flatten()(encoded)
    bottleneck = Dense(256, activation='relu')(flatten)

    # Reshape for Decoder
    reshape = Reshape((1000, 128))(bottleneck)

    # Decoder from seismonet_decoder.py
    decoded = build_decoder(reshape)

    # Model Compilation
    model = Model(inputs=input_signal, outputs=decoded)
    return model

# Train SeismoNet
def train_seismonet(ncs_path, stead_path):
    ncs_data = load_ncs_data(ncs_path, num_samples=5000)
    stead_data = load_stead_data(stead_path, num_samples=5000)
    dataset = np.concatenate((ncs_data, stead_data), axis=0)

    # Apply NSWF-Based Denoising Before Training
    dataset_denoised = np.array([apply_nswf_denoising(sig.flatten()) for sig in dataset])

    # Train-Test Split
    train_data, val_data = dataset_denoised[:8000], dataset_denoised[8000:]

    model = build_seismonet()
    model.compile(optimizer=BiDCG(learning_rate=0.0001), loss="mse", metrics=["mae"])

    # Learning Rate Scheduling and Early Stopping
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-8)
    early_stopper = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Train Model
    history = model.fit(
        train_data, train_data,
        validation_data=(val_data, val_data),
        batch_size=16, epochs=100,
        callbacks=[lr_scheduler, early_stopper],
        verbose=1
    )

    # Save Trained Model
    model.save(MODEL_PATH)

    # Plot Training Results
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss (MSE)")
    plt.title("Training and Validation Loss of SeismoNet")
    plt.legend()
    plt.show()

    return model

# Evaluate Performance
def evaluate_model(model, dataset):
    predictions = model.predict(dataset)

    # Compute Metrics
    mse = np.mean((dataset.flatten() - predictions.flatten()) ** 2)
    pcc = np.corrcoef(dataset.flatten(), predictions.flatten())[0, 1]

    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print(f"Pearson Correlation Coefficient (PCC): {pcc:.4f}")

# Run Training and Evaluation
if __name__ == "__main__":
    if not os.path.exists(STEAD_FILE):
        download_stead()
    if not os.path.exists(NCS_FILE):
        download_ncs()

    trained_model = train_seismonet(NCS_FILE, STEAD_FILE)
    test_dataset = load_ncs_data(NCS_FILE, num_samples=1000)
    evaluate_model(trained_model, test_dataset)
