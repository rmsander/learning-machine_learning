import numpy as np

# X_train and Y_train - Arrays to be saved
X_train = np.random.normal(size=(1000, 10))
Y_train = np.random.normal(size=(1000, 1))

# Save features X_train
np.save('x_train.npy', X_train)

# Save labels Y_train
np.save('y_train.npy', Y_train)

# Now try loading features and labels
X_train_reloaded = np.load('x_train.npy')
Y_train_reloaded = np.load('y_train.npy')

# Check for equality
print("X_train = X_train Reloaded? {}".format(np.all(np.equal(X_train, X_train_reloaded))))
print("Y_train = Y_train Reloaded? {}".format(np.all(np.equal(Y_train, Y_train_reloaded))))
