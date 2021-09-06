"""Function to compute stability metrics for covariance matrices,
namely log determinants and condition numbers."""
# Use torch
import torch

def compute_covariance_metrics(Kxx, x_train, writer):
    """Helper function to compute covariance metrics.

    Parameters:
        Kxx (torch.Tensor): Tensor object corresponding to a covariance matrix.
        writer (torch.utils.tensorboard.SummaryWriter): A summary writer object
            for logging metrics to tensorboard.
    """
    # Compute the condition number, p=2 norm gives \sigma_max / \sigma_min
    condition_number = torch.linalg.cond(Kxx, p=2)

    # Compute the log determinant as a stability metric
    log_det = torch.logdet(Kxx)

    # Compute the mean and variance of each metric, since models are batched
    vals = [condition_number, log_det]
    names = ["Condition Number", "Log Determinant"]
    for val, name in zip(vals, names):  # Loop jointly

        # Count number of NaNs
        num_nans = torch.sum(torch.isnan(val))

        # Get binary mask of indices where not NaN
        not_nan = ~(torch.isnan(val))

        # Get mean, maximum, and minimum
        mean_val = torch.mean(val[not_nan])
        max_val = torch.max(val[not_nan])
        min_val = torch.min(val[not_nan])
        types = [mean_val, max_val, min_val]
        name_types = ["Mean", "Max", "Min"]
