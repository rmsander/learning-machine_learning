"""Adapted from: https://colab.research.google.com/github/tensorflow/datasets/blob/master/docs/overview.ipynb#scrollTo=TTBSvHcSLBzc
"""

import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

ds = tfds.load('mnist', split='train')
ds = ds.take(-1)  # Only take a single example

# Number of examples
N = tf.data.experimental.cardinality(ds)

# Initialize output arrays for X and Y
X_train = np.zeros((N, 28, 28, 1))
Y_train = np.zeros((N,))

# Loop over each batch
for i, example in enumerate(ds):  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
  X_train[i] = example["image"].numpy()
  Y_train[i] = example["label"].numpy()
