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
X_train = None
Y_train = None

# Loop over each batch
for i, example in enumerate(ds):  # example is `{'image': tf.Tensor, 'label': tf.Tensor}`
  if X_train is None:
    X_train = example[0].numpy()
    Y_train = example[1].numpy()

  else:  # Concatenate
    X_train = np.concatenate((X_train, example[0].numpy()))    
    Y_train = np.concatenate((Y_train, example[1].numpy()))  
