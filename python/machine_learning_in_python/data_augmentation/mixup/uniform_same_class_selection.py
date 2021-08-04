import numpy as np
import tensorflow as tf

def vector_same_class_sample(X, Y, y_sample, num_classes=2):
  """Function for generating equal-class Mixup pairs in a vectorized fashion.
  Essentially, what we'll do is separate X into different sets according to the
  class of the corresponding index in Y. Then, given y_sample, a set of classes
  of the first set of images/vectors that we use for Mixup, we pair the same classes
  together appropriately by only sampling corresponding same-class images/vectors
  from X using appropriate indexing.
  Parameters:
    X (tf.Tensor): TensorFlow tensor representing the input values used for
      prediction. Expects shape (N, W, H, C), where N is the total dataset size, W is
      the width, H is the height, and C is the channels if we are using images, or
      (N, D) if using vectors, where D is the dimensionality of the vector.
    Y (tf.Tensor): TensorFlow tensor representing the target values used for
      training the model. Expects shape (N, 1) if sing binary classification, and
      (N, M) if multiclass classification, where M is the number of channels.
    y_sample (tf.Tensor): A sample of labels of shape (B, 1) - binary classification
      or (B, M) - multiclass classification, where B is the batch size and M is
      the number of classes considered.
  Returns:
    X_sample (tf.Tensor): A batch of inputs corresponding to the second set of
      inputs to be used for Mixup. This should have shape (B,) + X.shape.
  """
  # Create as NumPy arrays for easier manipulation
  X_np = X.numpy()
  Y_np = Y.numpy()
  y_sample_np = y_sample.numpy()

  # Create datasets
  D_idx = {}
  for i in range(num_classes):

    # Now take "slice of dataset" using these indices
    D_idx[i] = np.argwhere(Y_np == i)

  # Initalize output array
  output_shape = (y_sample.numpy().shape[0],) + X_np.shape[1:]
  X_sample = np.zeros(output_shape)

  # Now sample from datasets
  for j in range(num_classes):

    # Classes from sample
    idx_class_sample = np.argwhere(y_sample_np == j).reshape(-1)  # Samples where y = j
    N = idx_class_sample.size  # Number of these we need to sample from X

    # Indices to sample from
    idx_choice = D_idx[j].reshape(-1)  # Look up eligible indices

    # Now sample this many samples from X
    sampled_idx = np.random.choice(idx_choice, size=N, replace=True)  # Sample uniformly (iid) from "eligible" indices
    X_sample[idx_class_sample] = X_np[sampled_idx]  # Set output array for matching labels with sampled "eligible" indices

  return tf.convert_to_tensor(X_sample)

# Test

# Create datasets
Y1 = tf.convert_to_tensor(np.zeros(500))
Y2 = tf.convert_to_tensor(np.ones(500))
Y3 = 2 * tf.convert_to_tensor(np.ones(500))
Y = tf.concat([Y1, Y2, Y3], axis=0)

X1 = tf.convert_to_tensor(np.zeros((500, 10)))
X2 = tf.convert_to_tensor(np.ones((500, 10)))
X3 = 2 * tf.convert_to_tensor(np.ones((500, 10)))
X = tf.concat([X1, X2, X3], axis=0)

# Now randomly sample from X and Y
idx_sample = np.random.choice(1500, size=32).reshape((-1, 1))
y_sample = tf.gather_nd(Y, idx_sample)

# Now run function above
X_sample = vector_same_class_sample(X, Y, y_sample, num_classes=3)

print("Y SAMPLE: {}".format(y_sample))

print("X SAMPLE: {}".format(X_sample))

# Should pass this test
assert np.all(np.mean(X_sample, axis=1) == y_sample)
print("TEST PASSED!")
