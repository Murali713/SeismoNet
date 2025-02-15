import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import joblib
from tqdm import tqdm
from obspy import read
import h5py
import requests
from bidcg_optimizer import BiDCG  # Custom optimizer
from seismonet_decoder import build_decoder  # Decoder function
from sesimonet_encoder import build_encoder  # Encoder function

# Define dataset URLs
STEAD_URL = "https://www.kaggle.com/datasets/isevilla/stanford-earthquake-dataset-stead"
NCS_URL = "https://seismo.gov.in/data-portal"

# Directory to store datasets
DATA_DIR = "seismonet_data"
STEAD_FILE = os.path.join(DATA_DIR, "stead_dataset.hdf5")
NCS_FILE = os.path.join(DATA_DIR, "ncs_seismic_data.mseed")

# Ensure dataset directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Function to download STEAD dataset
def download_stead():
    print(f"Please download the STEAD dataset manually from: {STEAD_URL}")
    print(f"Once downloaded, place the file inside {DATA_DIR} with the name 'stead_dataset.hdf5'.")

# Function to download NCS dataset
def download_ncs():
    print(f"Please request access to the NCS dataset from: {NCS_URL}")
    print(f"Once obtained, place the file inside {DATA_DIR} with the name 'ncs_seismic_data.mseed'.")

# Load trained wavelet (NSWF)
def load_trained_wavelet():
    return joblib.load("trained_wavelet_nswf.pkl")  # Ensure this file exists

# Load NCS India Dataset
def load_ncs_data(ncs_file_path, num_samples=5000):
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
def load_stead_data(stead_file_path, num_samples=5000):
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

# Build Full SeismoNet Model (Encoder-Decoder)
def build_seismonet():
    input_signal = Input(shape=(4000, 1))

    # Encoder from sesimonet_encoder.py
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
    
    train_data, val_data = dataset[:8000], dataset[8000:]

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
    model.save("seismonet_trained.h5")

    # Plot Training Results
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss")
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
