import numpy as np
import matplotlib.pyplot as plt
from Custom_Wavelet_NSWF import custom_wavelet  # Import your wavelet function

# Step 1: Initialize Weights using NSWF
def initialize_weights(N, fs=100):
    """
    Initializes weights as wavelet coefficients using NSWF.

    Parameters:
        N (int): Number of coefficients.
        fs (int): Sampling frequency.

    Returns:
        w (array): Initialized weight vector.
    """
    n = np.arange(-N // 2, N // 2)
    wavelet_values = custom_wavelet(n, fs=fs)
    return np.real(wavelet_values)  # Use real part for initialization

# Step 2: Compute Loss Function and Gradient
def compute_loss(w, x, mu, k, sigma_local, lambda_reg):
    """
    Computes the loss function.

    Parameters:
        w (array): Current weight vector.
        x (array): Observed data.
        mu (float): Mean of wavelet coefficients.
        k (float): Scaling factor.
        sigma_local (float): Local variance.
        lambda_reg (float): Regularization weight.

    Returns:
        loss (float): Computed loss value.
    """
    predicted = w - (mu + k * sigma_local) + lambda_reg * w
    loss = np.mean((predicted - x) ** 2)
    return loss

def compute_gradient(w, x, mu, k, sigma_local, lambda_reg):
    """
    Computes the gradient of the loss function.

    Parameters:
        w (array): Current weight vector.
        x (array): Observed data.
        mu (float): Mean of wavelet coefficients.
        k (float): Scaling factor.
        sigma_local (float): Local variance.
        lambda_reg (float): Regularization weight.

    Returns:
        gradient (array): Computed gradient.
    """
    return 2 * (w - (mu + k * sigma_local) + lambda_reg * w - x)

# Step 3: Bi-Directional Conjugate Gradient (BiDCG) Optimizer
def bidirectional_cg_optimizer(x, N, alpha=0.01, epsilon=1e-6, max_iter=100, fs=100):
    """
    Implements the Bi-Directional Conjugate Gradient (BiDCG) optimizer with NSWF.

    Parameters:
        x (array): Observed data.
        N (int): Number of coefficients.
        alpha (float): Learning rate.
        epsilon (float): Convergence threshold.
        max_iter (int): Maximum iterations.
        fs (int): Sampling frequency for NSWF.

    Returns:
        w (array): Optimized weight vector.
        loss_history (list): Loss evolution over iterations.
    """
    # Initialize weights with NSWF
    w = initialize_weights(N, fs)
    mu = np.mean(w)
    sigma_local = np.std(w)
    lambda_reg = 0.001  # Regularization parameter

    loss_history = []
    gradient = compute_gradient(w, x, mu, k=1.5, sigma_local=sigma_local, lambda_reg=lambda_reg)
    d = -gradient
    prev_gradient = gradient.copy()

    for i in range(1, max_iter + 1):
        # Compute Loss
        loss = compute_loss(w, x, mu, k=1.5, sigma_local=sigma_local, lambda_reg=lambda_reg)
        loss_history.append(loss)

        # Check Convergence
        if np.linalg.norm(gradient) < epsilon:
            break

        # Compute β coefficients for Conjugate Gradient Update
        beta_FR = np.dot(gradient, gradient) / np.dot(prev_gradient, prev_gradient)
        beta_PR = np.dot((gradient - prev_gradient), gradient) / np.dot(prev_gradient, prev_gradient)

        # Compute angle between gradients
        theta_i = np.arccos(np.dot(gradient, prev_gradient) / (np.linalg.norm(gradient) * np.linalg.norm(prev_gradient)))

        # Moving Average of Previous Angles (Window Size: 5)
        if i > 5:
            theta_avg = np.mean(loss_history[-5:])
        else:
            theta_avg = theta_i

        # Choose Update Strategy
        if theta_i > theta_avg:
            d = -gradient + beta_FR * d
        else:
            d = -gradient + beta_PR * d

        # Adaptive Learning Rate Based on Wavelet Variance
        alpha = np.dot(gradient, d) / (np.dot(d, d) + np.var(w))

        # Update Weights
        w += alpha * d
        prev_gradient = gradient.copy()
        gradient = compute_gradient(w, x, mu, k=1.5, sigma_local=sigma_local, lambda_reg=lambda_reg)

    return w, loss_history

# Step 4: Loss Evolution Plot
def plot_loss_evolution(loss_history):
    """
    Plots the loss function evolution over training iterations.

    Parameters:
        loss_history (list): Loss values over iterations.

    Returns:
        None.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Loss Evolution")
    plt.xlabel("Iterations")
    plt.ylabel("Loss Value")
    plt.title("Loss Function Evolution with BiDCG and NSWF Regularization")
    plt.legend()
    plt.show()

# Step 5: Run BiDCG Optimization (Example Usage)
if __name__ == "__main__":
    N = 1024  # Number of wavelet coefficients
    x_observed = np.random.randn(N)  # Simulated seismic data

    optimized_w, loss_history = bidirectional_cg_optimizer(x_observed, N, fs=100)

    # Plot loss evolution
    plot_loss_evolution(loss_history)

    print(f"Final Optimized Weights: {optimized_w[:10]} (First 10 values)")
    print(f"Final Loss Value: {loss_history[-1]:.6f}")
