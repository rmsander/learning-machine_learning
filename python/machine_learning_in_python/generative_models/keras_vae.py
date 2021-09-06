"""Adapted from the Keras VAE guide: https://keras.io/examples/generative/vae/."""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def make_encoder():
  """Function for making the encoder."""
  latent_dim = 10

  encoder_inputs = keras.Input(shape=(316, 256, 1))
  x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
  x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2D(32, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2D(
  x = layers.Flatten()(x)
  x = layers.Dense(16, activation="relu")(x)
  z_mean = layers.Dense(latent_dim, name="z_mean")(x)
  z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
  encoder.summary()


def make_decoder():
  """Function for making the decoder."""
  latent_inputs = keras.Input(shape=(latent_dim,))
  x = layers.Dense(20*16*64, activation="relu")(latent_inputs)
  x = layers.Reshape((20, 16, 64))(x)
  x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
  x = layers.Conv2DTranspose(32, 3, activation="relu", strides=2, padding="same")(x)
  decoder_outputs = layers.Conv2DTranspose(1, 3, activation="sigmoid", padding="same")(x)
  decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")
  decoder.summary()
  return decoder
