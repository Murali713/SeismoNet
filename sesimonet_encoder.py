import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.models import Model
import numpy as np

# Step 1: Define Custom Wavelet Convolution Layer
class WaveletConv1D(tf.keras.layers.Layer):
    """
    Custom 1D Wavelet Convolution Layer using a Non-Standard Wavelet Function (NSWF).
    """
    def __init__(self, filters, kernel_size, strides=1, padding="same"):
        super(WaveletConv1D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel = self.add_weight(name="wavelet_kernel",
                                      shape=(kernel_size, 1, filters),
                                      initializer="glorot_uniform",
                                      trainable=True)

    def call(self, inputs):
        return tf.nn.conv1d(inputs, self.kernel, stride=self.strides, padding=self.padding.upper())

# Step 2: Define Adaptive Threshold Activation Function
def adaptive_thresholding(x, k=1.5):
    """
    Applies adaptive thresholding to the wavelet coefficients.
    """
    mean_x = tf.reduce_mean(x, axis=-1, keepdims=True)
    std_x = tf.math.reduce_std(x, axis=-1, keepdims=True)
    threshold = mean_x + k * std_x
    return tf.where(tf.abs(x) > threshold, x, tf.zeros_like(x))

# Step 3: Build the SeismoNet Encoder
def build_seismonet_encoder(input_shape=(1024, 1)):
    """
    Builds the encoder structure of SeismoNet.

    Parameters:
        input_shape (tuple): Shape of the input seismic signal.

    Returns:
        model (tf.keras.Model): SeismoNet encoder model.
    """
    inputs = Input(shape=input_shape)

    # First Convolutional Layer (Wavelet Convolution)
    x = WaveletConv1D(filters=64, kernel_size=7, strides=1, padding="same")(inputs)
    x = tf.keras.layers.Activation(lambda x: adaptive_thresholding(x))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.2)(x)

    # Second Convolutional Layer (Wavelet Convolution)
    x = WaveletConv1D(filters=128, kernel_size=5, strides=1, padding="same")(x)
    x = tf.keras.layers.Activation(lambda x: adaptive_thresholding(x))(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = Dropout(0.3)(x)

    # Flatten Layer
    x = Flatten()(x)

    # Fully Connected Dense Layer
    x = Dense(256, activation="relu")(x)
    x = Dense(512, activation="relu")(x)

    # Encoder Model
    model = Model(inputs=inputs, outputs=x, name="SeismoNet_Encoder")
    return model

# Step 4: Build and Compile the Model
seismonet_encoder = build_seismonet_encoder()
seismonet_encoder.summary()
