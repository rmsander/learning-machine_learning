import tensorflow as tf
import numpy as np


def mixup_tensorflow(X1, X2, alpha=1.0):
  """Function for implementing Mixup using TensorFlow.
  Mixup interpolation occurs between corresponding indices
  of X1[i] and X2[i].

  Parameters:
      X1 (tf.Tensor): Tensor object representing the first
        set of inputs to use for Mixup. Expects shape (N, D), where
        N is the number of samples, and D is the rest of the dimensions.
      X2 (tf.Tensor): Tensor object representing the second
        set of inputs to use for Mixup. Expects shape (N, D), where
        N is the number of samples, and D is the rest of the dimensions.
      alpha (float): Mixup alpha value.
  """
  # Cast tensors to float32 type
  X1 = tf.cast(X1, tf.float32)
  X2 = tf.cast(X2, tf.float32)
  print("Input shape X1: {}".format(X1.shape))
  print("Input shape X2: {}".format(X2.shape))

  # Get shapes of array
  N = X1.shape[0]
  d = X1.shape[1:] # Could be tuple or integer
  print("N: {}".format(N))
  print("D: {}".format(d))

  # Sample Mixup coefficient to determine convex linear interpolation
  b = np.random.beta(alpha, alpha, size=N)

  # Tile the coefficients (has the same dimensions as the vectors of X)
  for r in d:
    b = np.repeat(b[..., np.newaxis], r, axis=-1)

  print("B shape: {}".format(b.shape))

  # Cast Mixup coefficients to tf.float32
  B = tf.cast(tf.convert_to_tensor(b), tf.float32)

  # Take 1-b of sampled Mixup coefficients over dimensions
  one_minus_B = tf.cast(tf.ones(B.shape), tf.float32) - B

  print("B SHAPE: {}".format(B.shape))
  print("1-B SHAPE: {}".format(one_minus_B.shape))

  # Check to make sure we "tiled" correctly
  print("b[0] mean: {}".format(np.mean(b[0])))
  print("1-b[0] mean: {}".format(np.mean(one_minus_B.numpy()[0])))

  # Interpolate using Mixup coefficients
  X_interp = tf.add(tf.multiply(B, X1),
                    tf.multiply(one_minus_B, X2))

  return X_interp

## Working example ##

# Create input tensors (vectors)
x1 = tf.convert_to_tensor(np.ones((100, 2)))
x2 = tf.convert_to_tensor(np.zeros((100, 2)))

# Get Mixup (vectors)
x_interp_vector = mixup_tensorflow(x1, x2, alpha=1.0)
print("Vector shape of x_interp: {}".format(x_interp_vector.shape))
print("X[0] vector: {}".format(x_interp_vector[0]))  # Should be the same as b[0] mean

# Create input tensors (images/matrices)
x3 = tf.convert_to_tensor(np.ones((100, 32, 32)))
x4 = tf.convert_to_tensor(np.zeros((100, 32, 32)))

# Get Mixup (images/matrices)
x_interp_matrix = mixup_tensorflow(x3, x4, alpha=1.0)
print("Matrix shape of x_interp: {}".format(x_interp_matrix.shape))
print("X[0] matrix: {}".format(x_interp_matrix[0])) # Should be the same as b[0] mean
