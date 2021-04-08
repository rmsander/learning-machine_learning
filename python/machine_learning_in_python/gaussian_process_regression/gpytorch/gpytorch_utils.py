"""Functions and utility functions for creating and training Gaussian Process Regressor
objects through GPyTorch."""

from gpytorch.models import IndependentModelList
from gpytorch.likelihoods import GaussianLikelihood, LikelihoodList, MultitaskGaussianLikelihood
from gpytorch.mlls import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from gpytorch.constraints import GreaterThan
from botorch.fit import fit_gpytorch_model

# PyTorch
import torch

# NumPy
import numpy as np

# Timing and math
import math
import gc

# Import models
from utils.gpytorch.gpr_models import BatchedGP, CompositeBatchedGP, MultitaskGPModel

from parameters import MIN_INFERRED_NOISE_LEVEL


def train_gpytorch_modellist(Zs, Ys, use_cuda=False, epochs=10, lr=0.1, thr=0):
    """Computes a Gaussian Process object using GPyTorch.  Rather than training
    in a batched fashion, this approach trains Gaussian Process Regressors as a list.

    Parameters:
        Zs (np.array): Array of inputs of expanded shape (B, N, XD), where B is
            the size of the minibatch, N is the number of data points in each
            GP (the number of neighbors we consider in IER), and XD is the
            dimensionality of the state-action space of the environment.
        Ys (np.array): Array of predicted values of shape (B, N, YD), where B is the
            size of the minibatch and N is the number of data points in each
            GP (the number of neighbors we consider in IER), and YD is the
            dimensionality of the state-reward space of the environment.
        use_cuda (bool): Whether to use CUDA for GPU acceleration with PyTorch
            during the optimization step.  Defaults to False.
        epochs (int):  The number of epochs to train the batched GPs over.
            Defaults to 10.
        lr (float):  The learning rate to use for the Adam optimizer to train
            the batched GPs.
        thr (float):  The mll threshold at which to stop training.  Defaults to 0.

    Returns:
        model (BatchedGP): A GPR model of BatchedGP type with which to generate
            synthetic predictions of rewards and next states.
        likelihood (GaussianLikelihood): A likelihood object used for training
            and predicting samples with the BatchedGP model.
    """
    # Create GP Models using a Modellist object
    likelihoods = [MultitaskGaussianLikelihood(
        num_tasks=Ys.shape[-1]) for i in range(Ys.shape[0])]
    models = [MultitaskGPModel(
        Zs[i], Ys[i], likelihoods[i], num_tasks=Ys.shape[-1]) for i in range(len(likelihoods))]
    likelihood = LikelihoodList(*[model.likelihood for model in models])

    # Create the aggregated model
    model = IndependentModelList(*models)

    # Create marginal log likelihood object
    mll = SumMarginalLogLikelihood(likelihood, model)

    # Ensure model and likelihood are trainable
    model.train()
    likelihood.train()

    # Send everything to GPU for training
    if use_cuda:
        model = model.cuda()
        likelihood.cuda()

    # Use the Adam optimizer
    optimizer = torch.optim.Adam([{'params': model.parameters()}], lr=lr)

    # Initialize loss to high value
    loss_val = math.inf
    i = 0
    max_iters = epochs

    # Optimization loop
    while loss_val > thr and i < max_iters:
        optimizer.zero_grad()
        output = model(*model.train_inputs)
        loss = -mll(output, model.train_targets)
        loss.backward()
        optimizer.step()

        # Increment i and update loss tracker
        i += 1
        loss_val = loss.item()

    return model, model.likelihood


def train_gp_batched_scalar(Zs, Ys, use_cuda=False, epochs=10,
                            lr=0.1, thr=0, use_ard=False, composite_kernel=False,
                            kernel='matern', mean_type='zero', matern_nu=2.5,
                            ds=None, global_hyperparams=False,
                            model_hyperparams=None, use_botorch=False,
                            use_lbfgs=False, use_priors=False,
                            est_lengthscales=False, cluster_heuristic_lengthscales=None,
                            lengthscale_constant=2.0, lengthscale_prior_std=0.1):
    """Computes a Gaussian Process object using GPyTorch. Each outcome is
    modeled as a single scalar outcome.

    Parameters:
        Zs (torch.tensor): Tensor of inputs of expanded shape (B, N, XD), where B is
            the size of the minibatch, N is the number of data points in each
            GP (the number of neighbors we consider in IER), and XD is the
            dimensionality of the state-action space of the environment.
        Ys (torch.tensor): Tesor of predicted values of shape (B, N, YD), where B is the
            size of the minibatch and N is the number of data points in each
            GP (the number of neighbors we consider in IER), and YD is the
            dimensionality of the state-reward space of the environment.
        use_cuda (bool): Whether to use CUDA for GPU acceleration with PyTorch
            during the optimization step.  Defaults to False.
        epochs (int):  The number of epochs to train the batched GPs over.
            Defaults to 10.
        lr (float):  The learning rate to use for the Adam optimizer to train
            the batched GPs.
        thr (float):  The mll threshold at which to stop training.  Defaults to 0.
        use_ard (bool):  Whether to use Automatic Relevance Determination (ARD)
            for the lengthscale parameter, i.e. a weighting for each input dimension.
            Defaults to False.
        composite_kernel (bool):  Whether to use a composite kernel that computes
            the product between states and actions to compute the variance of y.
            Defaults to False.
        kernel (str):  Type of kernel to use for optimization. Defaults to "
            "Matern kernel ('matern'). Other options include RBF
            (Radial Basis Function)/SE (Squared Exponential) ('rbf'), and RQ
            (Rational Quadratic) ('rq').
        mean_type (str): Type of mean function to use for Gaussian Process.
            Defaults to zero mean ('zero').  Other options: linear
            ('linear'), and constant ('constant').")
        matern_nu (float): Value in set if {1/2, 3/2, 5/2} that denotes the power
            to raise the matern kernel evaluation to.  Smaller values allow for
            greater discontinuity.  Only relevant if kernel is matern.  Defaults to
            2.5.
        ds (int): If using a composite kernel, ds specifies the dimensionality of
            the state.  Only applicable if composite_kernel is True.
        global_hyperparams (bool):  Whether to use a single set of hyperparameters
            over an entire model.  Defaults to False.
        model_hyperparams (dict):  A dictionary of hyperparameters to use for
            initializing a model.  Defaults to None.
        use_botorch (bool):  Whether to optimize with L-BFGS using Botorch.
        use_lbfgs (bool):  Whether to use second-order gradient optimization with
            the L-BFGS PyTorch optimizer.  Note that this requires a closure
            function, as defined below.
        use_priors (bool):  Whether to use prior distribution over lengthscale and
            outputscale hyperparameters.  Defaults to False.
        est_lengthscales (bool):  Whether to estimate the lengthscales of each
            cluster of neighborhood by finding the farthest point in each dimension.
            Defaults to False.
        cluster_heuristic_lengthscales (np.array):  If computed, an array of clustered
            lengthscales used for estimation.
        matern_nu (float): Value in set if {1/2, 3/2, 5/2} that denotes the power
            to raise the matern kernel evaluation to.  Smaller values allow for
            greater discontinuity.  Only relevant if kernel is matern.  Defaults to
            2.5.
        lengthscale_constant (float):  Value which we multiply estimated lengthscales
            by.  Defaults to 1.0.
        lengthscale_prior_std (float):  Value for the standard deviation of the
            lengthscale prior.  Defaults to 0.1.

    Returns:
        model (BatchedGP): A GPR model of BatchedGP type with which to generate
            synthetic predictions of rewards and next states.
        likelihood (GaussianLikelihood): A likelihood object used for training
            and predicting samples with the BatchedGP model.
    """
    # Preprocess batch data
    B, N, XD = Zs.shape
    YD = Ys.shape[-1] if len(Ys.shape) > 2 else 1  # Note: this should always be > 2
    batch_shape = B * YD

    if use_cuda:  # If GPU available
        output_device = torch.device('cuda:0')  # GPU - todo(rms): generalize to multi-gpu
    else:
        output_device = torch.device('cpu')

    # Perform tiling/reshaping
    if YD > 1:
        train_x = Zs.repeat((YD, 1, 1)).double()  # Tile dimensions of x
        train_y = torch.vstack([Ys[..., i] for i in range(YD)]).double()

    # No need to perform tiling
    else:
        train_x = Zs.double()
        train_y = torch.squeeze(Ys, -1).double()

    # initialize likelihood and model
    likelihood = GaussianLikelihood(batch_shape=torch.Size([batch_shape]),
                                    noise_prior=None, noise_constraint=GreaterThan(MIN_INFERRED_NOISE_LEVEL))

    # Determine which type of kernel to use
    if composite_kernel:
        model = CompositeBatchedGP(train_x, train_y, likelihood, batch_shape,
                                   output_device, use_ard=use_ard, ds=ds,
                                   use_priors=use_priors, kernel=kernel,
                                   mean_type=mean_type, matern_nu=matern_nu,
                                   heuristic_lengthscales=cluster_heuristic_lengthscales)
    else:
        model = BatchedGP(train_x, train_y, likelihood, batch_shape,
                          output_device, use_ard=use_ard, use_priors=use_priors,
                          kernel=kernel, mean_type=mean_type, matern_nu=matern_nu,
                          heuristic_lengthscales=cluster_heuristic_lengthscales,
                          lengthscale_prior_std=lengthscale_prior_std)

    # Initialize the model with hyperparameters
    if model_hyperparams is not None:

        # Set model to eval mode
        model.initialize(**model_hyperparams)

    # Determine if we need to optimize hyperparameters
    if global_hyperparams:
        if use_cuda:  # Send everything to GPU for training

            # Make sure these are in "posterior" mode
            model.eval()
            likelihood.eval()

            # Empty the cache from GPU
            torch.cuda.empty_cache()
            gc.collect()  # NOTE: Critical to avoid GPU leak

            # Put model onto GPU
            model = model.cuda()
            likelihood = likelihood.cuda()

            del train_x, train_y, Zs, Ys
        return model, likelihood

    # Determine which optimizer to use
    if use_lbfgs:
        opt = torch.optim.LBFGS
        lr = 1.0
        epochs = 2
    else:
        opt = torch.optim.Adam

    # Model in prior mode
    model.train()
    likelihood.train()
    optimizer = opt(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    # Send everything to GPU for training
    if use_cuda:
        model = model.cuda().double()  # Convert model to float64
        likelihood = likelihood.cuda().double()  # Convert likelihood to float64
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        mll = mll.cuda()

    def closure():
        """Helper function, specifically for L-BFGS."""
        optimizer.zero_grad()
        output = model(train_x)  # Forwardpass
        loss = -mll(output, train_y).sum()  # Compute ind. losses + aggregate
        loss.backward()  # Backpropagate gradients
        item_loss = loss.item()  # Extract loss (detached from comp. graph)
        gc.collect()  # NOTE: Critical to avoid GPU leak
        return loss


    def epoch_train(j):
        """Helper function for running training in the optimization loop.  Note
        that the model and likelihood are updated outside of this function as well.

        Parameters:
            j (int):  The epoch number.

        Returns:
            item_loss (float):  The numeric representation (detached from the
                computation graph) of the loss from the jth epoch.
        """
        optimizer.zero_grad()  # Zero gradients
        output = model(train_x)  # Forwardpass
        loss = -mll(output, train_y).sum()  # Compute ind. losses + aggregate
        loss.backward()  # Backpropagate gradients
        item_loss = loss.item()  # Extract loss (detached from comp. graph)
        optimizer.step()  # Update weights
        optimizer.zero_grad()  # Zero gradients
        gc.collect()  # NOTE: Critical to avoid GPU leak
        return item_loss


    # Optimize with Botorch
    if use_botorch:
        fit_gpytorch_model(mll)
        gc.collect()

    # Optimize with ADAM or L-FBGS
    else:
        patience_count = 0
        losses = [math.inf]
        prev_loss = math.inf
        # Run the optimization loop
        for i in range(epochs):
            if use_lbfgs:  # Use L-BFGS
                loss = optimizer.step(closure).item()
                print("LOSS {}: {}".format(i, loss))
            else:  # Use ADAM
                loss_i = epoch_train(i)
                if i % 10 == 0:
                    print("LOSS EPOCH {}: {}".format(i, loss_i))
                if loss_i > prev_loss: #losses[max(i-GP_LAG, 0)]:
                    patience_count += 1
                    print("EARLY STOPPING")
                    break
                #if patience_count >= GP_PATIENCE:  # If we reach a certain loss threshold, stop training
                prev_loss = loss_i
                losses.append(loss_i)
            gc.collect()

    # Empty the cache from GPU
    torch.cuda.empty_cache()
    return model, likelihood


def preprocess_eval_inputs(Zs, d_y, device="cpu"):
    """Helper function to preprocess inputs for use with training
    targets and evaluation.

    Parameters:
        Zs (np.array): Array of inputs of expanded shape (B, N, XD), where B is
            the size of the minibatch, N is the number of data points in each
            GP (the number of neighbors we consider in IER), and XD is the
            dimensionality of the state-action space of the environment.
        d_y (int):  The dimensionality of the targets of GPR.
        device (str):  Device to output the tensor to.

    Returns:
        eval_x (torch.tensor):  Torch tensor of shape (B * YD, N, XD).  This
            tensor corresponding to a tiled set of inputs is used as input for
            the inference model in FP32 format.
    """
    # Preprocess batch data
    eval_x = torch.tensor(Zs, device=device).double()
    eval_x = eval_x.repeat((d_y, 1, 1))
    return eval_x


def standardize(Y):
    """
    Standardizes to zero-mean gaussian.  Preserves batch and output dimensions.

    Parameters:
        Y (torch.tensor): Tensor corresponding to the data to be standardized.
            Expects shape (batch_shape, num_samples, dim_y) if y is multidimensional
            or (batch_shape, num_samples) if y is one dimension.

    Returns:
        Y_norm (torch.tensor):  Standard-normal standardized data.  Shape is
            preserved.
        Y_std (torch.tensor): Standard deviation of the dataset across a given
            dimension.  This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.
        Y_mean (torch.tensor):  Mean of the dataset across a given dimension.
            This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.
    """
    stddim = -1 if Y.dim() < 2 else -2
    Y_std = Y.std(dim=stddim, keepdim=True)
    Y_std = Y_std.where(Y_std >= 1e-9, torch.full_like(Y_std, 1.0))
    Y_mean = Y.mean(dim=stddim, keepdim=True)
    return (Y - Y_mean / Y_std), Y_std, Y_mean

def unstandardize(Y, Y_std, Y_mean):
    """Unstandardizes outputs.  Relies on having pre-computed standard deivations
    and mean moments.

    Parameters:
        Y (torch.tensor): Tensor corresponding to the data to be standardized.
            Expects shape (batch_shape, num_samples, dim_y) if y is multidimensional
            or (batch_shape, num_samples) if y is one dimension.
        Y_std (torch.tensor): Standard deviation of the dataset across a given
            dimension.  This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.
        Y_mean (torch.tensor):  Mean of the dataset across a given dimension.
            This dimension is across each batch such that the moments
            (and if applicable, the dimension) are preserved and independent.

    Returns:
        Y (torch.tensor):  Unstandardized data, using the previously-computed moments.
          Shape is preserved.
    """
    tile_dim = Y.size()[-2]
    Y_std_tiled = Y_std.repeat((1,tile_dim, 1))
    Y_mean_tiled = Y_mean.repeat((1,tile_dim, 1))
    return torch.multiply(Y_std_tiled, Y) + Y_mean_tiled


def compute_dist(X):
    """Utility function for computing the furthest distance from X.

    Parameters:
        X (np.array):  Array of shape (N, D), where N is the number of samples,
            and D is the dimensionality of the data.  Note that X[0, :] should
            correspond to the center sample of the dataset.

        dist (np.array):  An array of distances corresponding to the maximum
            distance from each center to its furthest points.  Each element of
            this array corresponds to a different distance value.
    """
    x = X[:, 0:1, :]  # Center point
    K = X[:, 1:, :]  # Neighborhood
    return torch.max(torch.abs(torch.subtract(x, K)), axis=1).values


def compute_avg_dist(X):
    """Utility function for computing the average distance from X.

    Parameters:
        X (np.array):  Array of shape (B, N, D), where N is the number of samples,
            and D is the dimensionality of the data.  Note that X[0, :] should
            correspond to the center sample of the dataset.

        dist (np.array):  An array of distances corresponding to the maximum
            distance from each center to its furthest points.  Each element of
            this array corresponds to a different distance value.
    """
    x = X[:, 0:1, :]  # Center point
    K = X[:, 1:, :]   # Neighborhood

    if hasattr(X, "numpy"):  # Torch tensors
        return torch.mean(torch.abs(torch.subtract(x, K)), axis=1)
    elif hasattr(X, "shape"):  # NumPy arrays
        return np.mean(np.abs(np.subtract(x, K)), axis=1)


def heuristic_hyperparams(train_x, dy, lengthscale_constant=2.0, mc_clustering=False):
    """Function to estimate hyperparameters for use as global hyperparameter priors.
    Specifically, approximates the lengthscales for setting a prior by computing
    the average mean distance over clusters in each dimension.

    Parameters:
        train_x (torch.tensor):  Tensor of inputs of shape (B, N, D), where B
            is the batch size, N is the number of points, and D is the dimension
            of the data.
        dy (int): The dimension of the outputs.
        lengthscale_constant (float): The value we multiply the mean estimates for
            the lengthscales.
        mc_clustering (bool): Whether or not the parameters presented are
                derived from a set of clusters.
    """
    # Compute mean distances
    x = train_x[:, 0:1, :]  # Center point
    K = train_x[:, 1:, :]  # Neighborhood
    mean_dist_over_clusters = np.mean(np.abs(np.subtract(x, K)), axis=1)

    # Now tile as needed
    if mc_clustering:  # Use clustered data
        scaled_lengthscales = lengthscale_constant * torch.tensor(mean_dist_over_clusters)
        cluster_heuristic_lengthscales = torch.unsqueeze(scaled_lengthscales, 1)
        cluster_heuristic_lengthscales = cluster_heuristic_lengthscales.repeat((dy, 1, 1)).double()
        return cluster_heuristic_lengthscales

    else:  # Use non-clustered data
        aggregate_mean_dist = np.mean(mean_dist_over_clusters, axis=0)
        tiled_aggregate_mean_dist = torch.tensor(np.tile(aggregate_mean_dist, (dy, 1)))
        return lengthscale_constant * tiled_aggregate_mean_dist


def format_preds(Y, B, single_model=False):
    """Function to format a tensor outputted from a GPR, either by tiling over
    intervals or transposition.

    Parameters:
        Y (torch.tensor):  Tensor corresponding to predicted outputs.  If:

            1. single_model is False, expects a tensor of shape (B * Yd, 1), where B
            is the batch size, and Yd is the dimension of the output.  Outputs of
            the ith element of the batch are given as i + B*j, where j is the
            dimension beginning at 0.

            2. single_model is True, expects a tensor of shape (Yd, B), where B
            is the batch size and Yd is the output. In this case, transposition
            is performed.

        B (int): The batch size (used for splitting).

        single_model (bool):  Whether a single model is used for prediction, or
            prediction is made in a batched format.

    Returns:
        Y_tiled (torch.tensor):  An output tensor of shape (B, Yd), where B is
        the batch size, and Yd is the dimension of the outputs.  Each row corresponds
        to an output, and each column corresponds to a different predicted feature.
    """
    # Reformat the likelihoods to compute weights
    if single_model:
        return torch.transpose(Y, 0, 1)
    else:
        return torch.squeeze(torch.stack(torch.split(Y, B, dim=0), 1))


def format_preds_queue(Y, B, single_model=False):
    """Function to format a tensor outputted from a GPR, either by tiling over
    intervals or transposition.  Allows for repetition with test points.

    Parameters:
        Y (torch.tensor):  Tensor corresponding to predicted outputs.  If:

            1. single_model is False, expects a tensor of shape (B * Yd, 1), where B
            is the batch size, and Yd is the dimension of the output.  Outputs of
            the ith element of the batch are given as i + B*j, where j is the
            dimension beginning at 0.

            2. single_model is True, expects a tensor of shape (Yd, B), where B
            is the batch size and Yd is the output. In this case, transposition
            is performed.

        B (int): The batch size (used for splitting).

        single_model (bool):  Whether a single model is used for prediction, or
            prediction is made in a batched format.

    Returns:
        Y_tiled (torch.tensor):  An output tensor of shape (B, Yd), where B is
        the batch size, and Yd is the dimension of the outputs.  Each row corresponds
        to an output, and each column corresponds to a different predicted feature.
    """
    # Reformat the likelihoods to compute weights
    if single_model:
        return torch.transpose(Y, 0, 1)
    else:
        return torch.squeeze(torch.stack(torch.split(Y, B, dim=0), -1))
