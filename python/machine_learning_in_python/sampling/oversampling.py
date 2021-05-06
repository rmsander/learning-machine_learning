import numpy as np


# Get labels and features from tf.dataset
X_train = np.concatenate([x for x, y in train_ds], axis=0)
Y_train = np.concatenate([y for x, y in train_ds], axis=0)

# Get number of positives and imbalance ratio
num_positive_examples = np.sum(Y_train)
num_negative_examples = Y_train.size - num_positive_examples
print("Imbalance ratio, 0: 1 --> {}".format(num_positive_examples / num_negative_examples))

# Get random indices for positive
idx_negative = np.where(Y_train == 0)[0]
subsample_idx_negative = np.random.choice(idx_negative, num_positive_examples)

# Get negative subset from resampling
X_train_neg = X_train[subsample_idx_negative, ...]
Y_train_neg = Y_train[subsample_idx_negative, ...]

# Get positive subset
idx_positive = np.where(Y_train == 1)[0]
X_train_pos = X_train[idx_positive, ...]
Y_train_pos = Y_train[idx_positive, ...]

# Now merge the two datasets and shuffle them
X_train = np.concatenate((X_train_neg, X_train_pos))
Y_train = np.concatenate((Y_train_neg, Y_train_pos))

# Shuffle the data
from sklearn.utils import shuffle
X_train, Y_train = shuffle(X_train, Y_train)

print("X SHAPE: {}".format(X_train.shape))
print("Y SHAPE: {}".format(Y_train.shape))
