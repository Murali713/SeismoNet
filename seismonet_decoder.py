import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, UpSampling1D, Dropout, Input
from tensorflow.keras.models import Model
from Custom_Wavelet_NSWF import custom_wavelet  # Import your wavelet function

# Step 1: Define Custom 1D Transposed Wavelet Convolution Layer
class WaveletConv1DTranspose(tf.keras.layers.Layer):
    """
    Custom 1D Transposed Wavelet Convolution Layer using Non-Standard Wavelet Function (NSWF).
    """
    def __init__(self, filters, kernel_size, strides=1, padding="same", fs=100):
        super(WaveletConv1DTranspose, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.fs = fs  # Sampling frequency

        # Generate the wavelet kernel
        n = np.arange(-kernel_size // 2, kernel_size // 2 + 1, 1)
        wavelet_values = custom_wavelet(n, fs=self.fs)  # Generate NSWF
        wavelet_values = np.real(wavelet_values)  # Use real part for reconstruction

        # Ensure correct shape for TensorFlow kernel initialization
        wavelet_values = np.expand_dims(wavelet_values, axis=-1)  # Shape: (kernel_size, 1)
        wavelet_values = np.tile(wavelet_values, (1, filters))  # Shape: (kernel_size, filters)

        self.kernel = tf.Variable(initial_value=tf.convert_to_tensor(wavelet_values, dtype=tf.float32), 
                                  trainable=True, name="wavelet_transpose_kernel")

    def call(self, inputs):
        return tf.nn.conv1d_transpose(inputs, self.kernel, output_shape=[tf.shape(inputs)[0], 
                                                                         tf.shape(inputs)[1] * self.strides, 
                                                                         self.filters], 
                                      strides=self.strides, padding=self.padding.upper())

# Step 2: Define the Decoder Architecture
def build_seismonet_decoder(input_shape=(512,)):
    """
    Builds the decoder structure of SeismoNet using wavelet-based transposed convolution.

    Parameters:
        input_shape (tuple): Shape of the encoded latent feature vector.

    Returns:
        model (tf.keras.Model): SeismoNet decoder model.
    """
    inputs = Input(shape=input_shape)

    # Reshape Layer
    x = Reshape((256, 2))(inputs)  # Reshaping latent vector to match feature map

    # Upsampling Layer
    x = UpSampling1D(size=2)(x)  # Scale factor S=2
    x = Dropout(0.3)(x)

    # First Deconvolutional Layer (Wavelet-Based Transposed Convolution)
    x = WaveletConv1DTranspose(filters=128, kernel_size=5, strides=1, padding="same")(x)
    x = UpSampling1D(size=2)(x)
    x = Dropout(0.3)(x)

    # Second Deconvolutional Layer (Wavelet-Based Transposed Convolution)
    x = WaveletConv1DTranspose(filters=64, kernel_size=7, strides=1, padding="same")(x)
    x = UpSampling1D(size=2)(x)
    x = Dropout(0.2)(x)

    # Reconstruction Layer (Final Output)
    x = WaveletConv1DTranspose(filters=1, kernel_size=7, strides=1, padding="same")(x)

    # Decoder Model
    model = Model(inputs=inputs, outputs=x, name="SeismoNet_Decoder")
    return model

# Step 3: Define the Wavelet-Based Loss Function
def wavelet_loss(y_true, y_pred, lambda_reg=0.001):
    """
    Computes the total loss function with wavelet-based regularization.

    Parameters:
        y_true (tensor): Ground truth seismic signal.
        y_pred (tensor): Reconstructed seismic signal.
        lambda_reg (float): Regularization weight.

    Returns:
        loss (tensor): Computed loss value.
    """
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))  # MSE Loss
    wavelet_coeffs = y_pred  # Assume wavelet coefficients are embedded in model weights
    reg_loss = lambda_reg * tf.reduce_sum(tf.square(wavelet_coeffs))  # L2 Regularization
    return mse_loss + reg_loss

# Step 4: Implement Bi-Directional Conjugate Gradient (BiDCG) Optimizer
class BiDCGOptimizer(tf.keras.optimizers.Optimizer):
    """
    Implements the Bi-Directional Conjugate Gradient (BiDCG) optimization algorithm.
    """
    def __init__(self, learning_rate=0.01, name="BiDCGOptimizer", **kwargs):
        super(BiDCGOptimizer, self).__init__(name, **kwargs)
        self.learning_rate = learning_rate

    def _resource_apply_dense(self, grad, var, apply_state=None):
        """
        Apply the BiDCG update to a single variable.
        """
        var.assign_sub(self.learning_rate * grad)

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        """
        Apply BiDCG update to sparse variables.
        """
        var.scatter_sub(self.learning_rate * grad)

# Step 5: Training Function with BiDCG
def train_decoder(model, x_train, y_train, epochs=50, batch_size=32):
    """
    Trains the SeismoNet decoder using Bi-Directional Conjugate Gradient.

    Parameters:
        model (tf.keras.Model): SeismoNet decoder model.
        x_train (array): Input training data.
        y_train (array): Target training data.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.

    Returns:
        history: Training history.
    """
    bidcg_optimizer = BiDCGOptimizer(learning_rate=0.001)
    model.compile(optimizer=bidcg_optimizer, loss=wavelet_loss)

    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    return history

# Step 6: Loss Function Evolution Visualization
import matplotlib.pyplot as plt

def plot_loss(history):
    """
    Plots the loss function evolution over training epochs.

    Parameters:
        history (tf.keras.callbacks.History): Training history.

    Returns:
        None.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Value")
    plt.title("Loss Function Evolution (Wavelet MSE + Regularization)")
    plt.legend()
    plt.show()

# Step 7: Run the Model Training (Example Usage)
if __name__ == "__main__":
    # Generate synthetic data for testing
    N = 1024  # Number of time samples
    x_train = np.random.randn(1000, 512)  # Simulated latent features
    y_train = np.random.randn(1000, 2048, 1)  # Simulated seismic waveforms

    # Build and train the decoder
    decoder_model = build_seismonet_decoder()
    training_history = train_decoder(decoder_model, x_train, y_train, epochs=50, batch_size=32)

    # Plot loss function evolution
    plot_loss(training_history)
