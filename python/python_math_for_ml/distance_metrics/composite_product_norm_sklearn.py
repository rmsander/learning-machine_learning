"""Function for implementing custom distance metrics for use in sklearn neighbor
search algorithms.

Here, we will use a composite product norm composed of a product of norms over
given (contiguous) subsets of states. The norms over these subsets are taken
to be L2 norms.

If x = [s a]^T, where s and a are the contiguous subsets of x that this
product norm is decomposed into, then this product norm can be written as:

d(x1, x2) = ||s1 - s2||_2 x ||a_1 - a_2||_2
"""
# External Python packages
import numpy as np


def composite_product_norm(x1, x2, **kwargs):
    """Function defining a composite product norm over states and actions.

    This norm is intended to be used with GPR kernels that have a product
    decomposition over states and actions. This norm can therefore be used to
    calculate composite L2 distance between states and actions. It is therefore
    only small when similar actions are taken in similar states.

    Parameters:
        x1 (np.array): Array corresponding to the first input vector. Distance
            is calculated as the distance between x1 and x2.
        x2 (np.array): Array corresponding to the second input vector. Distance
            is calculated as the distance between x2 and x1.

    Returns:
        d (float): A float value corresponding to the composite product norm
            distance between x1 and x2 in standardized space. Since points are
            standardized in each dimension, this is a measure of the product of
            state and action similarities.
    """
    # Get state dimension
    ds = kwargs["ds"]

    # Get states and actions of both inputs
    x1_states, x1_actions = x1[:ds], x1[ds:]
    x2_states, x2_actions = x2[:ds], x2[ds:]

    # Now compute product of L2 norms
    return np.linalg.norm(x1_states - x2_states, ord=2) * \
           np.linalg.norm(x1_actions - x2_actions, ord=2)
