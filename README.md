SeismoNet: A Deep Learning Approach for Seismic Signal Denoising and Detection
Overview
SeismoNet is a deep learning framework designed for time-frequency analysis and denoising of seismic signals using a Non-Standard Wavelet Function (NSWF). It replaces conventional fixed filters with learnable wavelets, making it more effective in isolating seismic events from noisy components. The model is optimized using a Bi-Directional Conjugate Gradient (BiDCG) optimizer for better noise suppression and signal reconstruction.
Repository Structure:
This repository contains all the necessary files required for training, testing, and evaluating the SeismoNet model.
Before running the project, install the necessary dependencies:
Requirements include: pip install -r requirements.txt
TensorFlow/PyTorch
NumPy
SciPy
OpenCV
Matplotlib
Pandas
Seaborn
