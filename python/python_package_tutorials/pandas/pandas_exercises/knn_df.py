# # Find Nearest Neighbors of Tabular Data
# In this notebook, we compute the nearest neighbors of different row items in a tabular dataset (loaded as a `pd.DataFrame`) using K-Nearest Neighbors and FAISS trees.

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import faiss

# Parameters
K_NEIGHBORS = 2

# "Read file" (dummy data)
df = pd.DataFrame({c: np.random.randint(low=1, high=190, size=10000) for c in ['a', 'b']})
df #pd.read_csv(FILE)


# Weight Features With User Input
# Get size of data to provide user with weights
n_cols = df.values.shape[1]  # Get number of columns
print("Please enter {} weights, one for each column:".format(n_cols))
query = input("Please enter {} numbers in list format, e.g. {} \n -->".format(n_cols, [i for i in range(n_cols)]))
weights = np.array(eval(query))  # Reads as list, and converts to NumPy array
print("Weights are: {}".format(weights))


# Now "weight the features"
X = df.values  # Creates a NumPy array
X_norm = X * weights  # Scales values by weights


# Find Neighbors Using [K-Nearest Neighbor Objects](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html)
# KNN objects can be used to compute nearest neighbors using a variety of distance metrics, which can be helpful for instances where data has different scales (e.g. Mahalanobis distance), is gridded (Manhattan distance), or has other unique constraints.  Below, we will implement KNN objects with the `sklearn` Python libary.

# Create K-Nearest Neighbors object
knn = NearestNeighbors(n_neighbors=K_NEIGHBORS, metric="minkowski", p=2)
knn.fit(X_norm)  # Fit the knn object to the data
neighbor_idx = knn.kneighbors(X_norm, K_NEIGHBORS+1, return_distance=False)[:, 1:]  # Can't take yourself
neighbor_dict = {i: k for i, k in enumerate(neighbor_idx)}  # Indexed by X_norm

# Find Neighbors Using [FAISS Trees](https://github.com/facebookresearch/faiss)
# FAISS trees are especially helpful for running large queries.  These operations are run in C in order to improve runtime, and can result in signicantly faster queries for large datasets/high dimensions. Below, we'll show how to query neighbors using L2-norms (Euclidean distance), implemented with the `faiss` Python libary.

class FaissKNeighbors:
    """An implementation of FAISS trees.

    Parameters:
        k (int): The number of neighbors we consider for the FAISS tree.
    """
    def __init__(self, k=50):
        self.index = None
        self.k = k

    def fit(self, X):
        """Function to fit the FAISS tree.

        Parameters:
            X (np.array):  Array of shape (N, d), where N is the number of
                samples and d is the dimension of hte data.  Note that the
                array must be of type np.float32.
        """
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))

    def query(self, X, k=None):
        """Function to query the neighbors of the FAISS tree.

        Parameters:
            X (np.array):  Array of shape (N, D), where N is the number of
                samples, and D is the dimension of the features.
            k (int):  If provided, the number of neighbors to compute.  Defaults
                to None, in which case self.k is used as the number of neighbors.

        Returns:
            indices (np.array): Array of shape (N, K), where N is the number of
                samples, and K is the number of nearest neighbors to be computed.
                The ith row corresponds to the k-nearest neighbors of the ith
                sample.
        """
        # Set number of neighbors
        if k is None:  # Use default number of neighbors
            k = self.k

        # Query and return nearest neighbors
        _, indices = self.index.search(X.astype(np.float32), k=k)
        return indices

# Need to convert array to C-order
X_norm = X_norm.copy(order="C")

# Fit the FAISS tree
faiss_tree = FaissKNeighbors(k=K_NEIGHBORS+1)  # Creates the FAISS tree
faiss_tree.fit(X_norm)  # Fits using L2 norm (Euclidean distance)

# Now query neighbors of the FAISS tree
neighbor_idx = faiss_tree.query(X_norm)[:, 1:]
neighbor_dict = {i: k for i, k in enumerate(neighbor_idx)}  # Indexed by X_norm

def find_k_nearest(f_csv, kneighbors=5, use_faiss=False, metric="minkowski", p=2):
    
    # Read csv file into pandas DataFrame
    df = pd.read_csv(f_csv)    
    
    # Get size of data to provide user with weights
    n_cols = df.values.shape[1]  # Get number of columns
    print("Please enter {} weights, one for each column:".format(n_cols))
    query = input("Please enter {} numbers in list format, e.g. {} \n -->".format(n_cols, [i for i in range(n_cols)]))
    weights = np.array(eval(query))  # Reads as list, and converts to NumPy array
    print("Weights are: {}".format(weights))
    
    # Now "weight the features"
    X = df.values  # Creates a NumPy array
    X_norm = X * weights  # Scales values by weights
    
    # Use KNN
    if not use_faiss:
        # Create K-Nearest Neighbors object
        knn = NearestNeighbors(n_neighbors=kneighbors, metric=metric, p=p)
        knn.fit(X_norm)  # Fit the knn object to the data
        
        # Now query neighbors of the KNN tree (NOTE: closest neighbor may be point itself)
        neighbor_idx = knn.kneighbors(X_norm, kneighbors+1, return_distance=False)[:, 1:]
        neighbor_dict = {i: k for i, k in enumerate(neighbor_idx)}  # Indexed by X_norm
    
    # Use FAISS tree
    else:
        # Need to convert array to C-order
        X_norm = X_norm.copy(order="C")

        # Fit the FAISS tree
        faiss_tree = FaissKNeighbors(k=kneighbors+1)  # Creates the FAISS tree
        faiss_tree.fit(X_norm)  # Fits using L2 norm (Euclidean distance)

        # Now query neighbors of the FAISS trees
        neighbor_idx = faiss_tree.query(X_norm)[:, 1:]
        neighbor_dict = {i: k for i, k in enumerate(neighbor_idx)}  # Indexed by X_norm
    
    return neighbor_dict
