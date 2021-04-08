"""Class used for implementing faster nearest neighbor lookup."""

import numpy as np
import faiss

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