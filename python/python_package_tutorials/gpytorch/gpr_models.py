"""Classes and utility functions for GPyTorch GPR models."""

# GPyTorch
import gpytorch
from gpytorch.models import ExactGP, IndependentModelList
from gpytorch.means import ConstantMean, ZeroMean, MultitaskMean, LinearMean
from gpytorch.kernels import RQKernel, RBFKernel, RBFKernelGrad, MaternKernel, \
    ProductKernel, ScaleKernel, MultitaskKernel
from gpytorch.kernels.keops import RBFKernel, MaternKernel  # TODO(rms): Integrate KeOps
from gpytorch.distributions import MultivariateNormal, MultitaskMultivariateNormal
from gpytorch.priors import GammaPrior, LogNormalPrior, NormalPrior

# PyTorch
import torch

class GPModel(ExactGP):
    """Class for exact Gaussian Process inference.  Each of these corresponds
    to each of the points we sample for the replay buffer.

    Parameters:
        train_x (torch.tensor): The training features used for Gaussian Process
            Regression.  These features will take shape (N, XD), where:

            (ii) N is the number of data points per GPR - the neighbors considered
            (iii) XD is the dimension of the features (d_state + d_action)

        train_y (torch.tensor): The training labels used for Gaussian Process
            Regression.  These features will take shape (N), where:

            (ii) N is the number of data points per GPR - the neighbors considered

        likelihood (gpytorch.likelihoods.GaussianLikelihood): A likelihood object
            used for training and predicting samples with the BatchedGP model.
    """
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(MaternKernel())

    def forward(self, x):
        """Forward pass method for making predictions through the model.  The
        mean and covariance are each computed to produce a MV distribution.

        Parameters:
            x (torch.tensor): The tensor for which we predict a mean and
                covariance used the BatchedGP model.

        Returns:
            mv_normal (gpytorch.distributions.MultivariateNormal): A Multivariate
                Normal distribution with parameters for mean and covariance computed
                at x.
        """
        mean_x = self.mean_module(x)  # Compute the mean at x
        covar_x = self.covar_module(x)  # Compute the covariance at x
        return MultivariateNormal(mean_x, covar_x)


class MultitaskGPModel(ExactGP):
    """Class for multitask Gaussian Process.  Each of these points corresponds to
    a batch of points that we sample from the replay buffer."""
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = MultitaskMean(ConstantMean(), num_tasks=num_tasks)
        self.covar_module = MultitaskKernel(MaternKernel(), num_tasks=num_tasks)

    def forward(self, x):
        """Forward pass method for making predictions through the model.  The
        mean and covariance are each computed to produce a MV distribution.

        Parameters:
            x (torch.tensor): The tensor for which we predict a mean and
                covariance used the BatchedGP model.

        Returns:
            mv_normal (gpytorch.distributions.MultivariateNormal): A Multivariate
                Normal distribution with parameters for mean and covariance computed
                at x.
        """
        mean_x = self.mean_module(x)  # Compute the mean at x
        covar_x = self.covar_module(x)  # Compute the covariance at x
        return MultitaskMultivariateNormal(mean_x, covar_x)


class BatchedGP(ExactGP):
    """Class for creating batched Gaussian Process Regression models.  Ideal candidate if
    using GPU-based acceleration such as CUDA for training.

    Parameters:
        train_x (torch.tensor): The training features used for Gaussian Process
            Regression.  These features will take shape (B * YD, N, XD), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) XD is the dimension of the features (d_state + d_action)
                (iv) YD is the dimension of the labels (d_reward + d_state)
            The features of train_x are tiled YD times along the first dimension.

        train_y (torch.tensor): The training labels used for Gaussian Process
            Regression.  These features will take shape (B * YD, N), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) YD is the dimension of the labels (d_reward + d_state)
            The features of train_y are stacked.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): A likelihood object
            used for training and predicting samples with the BatchedGP model.
        shape (int):  The batch shape used for creating this BatchedGP model.
            This corresponds to the number of samples we wish to interpolate.
        output_device (str):  The device on which the GPR will be trained on.
        use_ard (bool):  Whether to use Automatic Relevance Determination (ARD)
            for the lengthscale parameter, i.e. a weighting for each input dimension.
            Defaults to False.
        use_priors (bool):  Whether to use prior distribution over lengthscale and
            outputscale hyperparameters.  Defaults to False.
        kernel (str):  Type of kernel to use for optimization. Defaults to "
            "Rational Quadratic ('rq'). Other options include RBF
            (Radial Basis Function)/SE (Squared Exponential) ('rbf'), and Matern
             ('matern').
        mean_type (str): Type of mean function to use for Gaussian Process.
            Defaults to zero mean ('zero').  Other options: linear
            ('linear'), and constant ('constant').")
        matern_nu (float): Value in set if {1/2, 3/2, 5/2} that denotes the power
            to raise the matern kernel evaluation to.  Smaller values allow for
            greater discontinuity.  Only relevant if kernel is matern.  Defaults to
            2.5.
        heuristic_lengthscales (torch.tensor):  A tensor corresponding to estimated
            lengthscales of each of the local neighborhoods.  Defaults to None.
        lengthscale_prior_std (float):  Value for the standard deviation of the
            lengthscale prior.  Defaults to 0.1.
    """
    def __init__(self, train_x, train_y, likelihood, shape, output_device,
                 use_ard=False, use_priors=False, kernel='rq',
                 mean_type='zero', matern_nu=2.5, heuristic_lengthscales=None,
                 lengthscale_prior_std=0.1):

        # Run constructor of superclass
        super(BatchedGP, self).__init__(train_x, train_y, likelihood)

        # Determine if using ARD
        ard_num_dims = None
        if use_ard:
            ard_num_dims = train_x.shape[-1]

        # Get input size
        input_size = train_x.shape[-1]
        self.shape = torch.Size([shape])

        # Get mean function and kernel
        M, K, kwargs = get_mean_and_kernel(kernel, mean_type, shape,
                                           input_size, is_composite=False,
                                           matern_nu=matern_nu)
        # Default priors work if targets standardized and features min-max normalized
        lengthscale_prior = None
        outputscale_prior = None

        # Now construct mean function and kernel
        self.mean_module = M
        self.base_kernel = K(batch_shape=self.shape,
                             ard_num_dims=ard_num_dims,
                             lengthscale_prior=lengthscale_prior,
                             **kwargs)
        self.covar_module = ScaleKernel(self.base_kernel,
                                        batch_shape=self.shape,
                                        output_device=output_device,
                                        outputscale_prior=outputscale_prior)

        # Set priors, if applicable
        if lengthscale_prior is not None:
            self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean

        elif use_priors:
            self.covar_module.base_kernel.lengthscale = lengthscale_prior.mean



    def forward(self, x):
        """Forward pass method for making predictions through the model.  The
        mean and covariance are each computed to produce a MV distribution.

        Parameters:
            x (torch.tensor): The tensor for which we predict a mean and
                covariance used the BatchedGP model.

        Returns:
            mv_normal (gpytorch.distributions.MultivariateNormal): A Multivariate
                Normal distribution with parameters for mean and covariance computed
                at x.
        """
        mean_x = self.mean_module(x)  # Compute the mean at x
        covar_x = self.covar_module(x)  # Compute the covariance at x
        return MultivariateNormal(mean_x, covar_x)


class CompositeBatchedGP(ExactGP):
    """Class for creating batched Gaussian Process Regression models.  Ideal candidate if
    using GPU-based acceleration such as CUDA for training.

    This kernel produces a composite kernel that multiplies actions times states,
    i.e. we have a different kernel for both the actions and states.  In turn,
    the composite kernel is then multiplied by a Scale kernel.

    Parameters:
        train_x (torch.tensor): The training features used for Gaussian Process
            Regression.  These features will take shape (B * YD, N, XD), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) XD is the dimension of the features (d_state + d_action)
                (iv) YD is the dimension of the labels (d_reward + d_state)
            The features of train_x are tiled YD times along the first dimension.

        train_y (torch.tensor): The training labels used for Gaussian Process
            Regression.  These features will take shape (B * YD, N), where:
                (i) B is the batch dimension - minibatch size
                (ii) N is the number of data points per GPR - the neighbors considered
                (iii) YD is the dimension of the labels (d_reward + d_state)
            The features of train_y are stacked.

        likelihood (gpytorch.likelihoods.GaussianLikelihood): A likelihood object
            used for training and predicting samples with the BatchedGP model.
        shape (int):  The batch shape used for creating this BatchedGP model.
            This corresponds to the number of samples we wish to interpolate.
        output_device (str):  The device on which the GPR will be trained on.
        use_ard (bool):  Whether to use Automatic Relevance Determination (ARD)
            for the lengthscale parameter, i.e. a weighting for each input dimension.
            Defaults to False.
        ds (int): If using a composite kernel, ds specifies the dimensionality of
            the state.  Only applicable if composite_kernel is True.
        kernel (str):  Type of kernel to use for optimization. Defaults to "
            "Matern kernel ('matern'). Other options include RBF
            (Radial Basis Function)/SE (Squared Exponential) ('rbf'), and RQ
            (Rational Quadratic) ('rq').
        mean_type (str): Type of mean function to use for Gaussian Process.
            Defaults to constant mean ('constant').  Other options: linear
            ('linear'), and zero ('zero').")
        matern_nu (float): Value in set if {1/2, 3/2, 5/2} that denotes the power
            to raise the matern kernel evaluation to.  Smaller values allow for
            greater discontinuity.  Only relevant if kernel is matern.  Defaults to
            2.5.
        heuristic_lengthscales (torch.tensor):  A tensor corresponding to estimated
            lengthscales of each of the local neighborhoods.  Defaults to None.
    """
    def __init__(self, train_x, train_y, likelihood, shape, output_device,
                 use_ard=False, ds=None, use_priors=False,
                 kernel='rq', mean_type='zero', matern_nu=2.5,
                 heuristic_lengthscales=None):

        # Run constructor of superclass
        super(CompositeBatchedGP, self).__init__(train_x, train_y, likelihood)

        # Check if ds is None, and if not, set
        if ds is None:
            raise Exception("No dimension for state specified.  Please specify ds.")
        self.ds = ds

        # Set active dimensions
        state_dims = torch.tensor([i for i in range(0, ds)])
        action_dims = torch.tensor([i for i in range(ds, train_x.shape[-1])])

        # Determine if using ARD
        state_ard_num_dims = None
        action_ard_num_dims = None
        if use_ard:
            state_ard_num_dims = ds
            action_ard_num_dims = train_x.shape[-1] - ds

        # Get input size for mean module
        input_size = train_x.shape[-1]

        # Create the mean and covariance modules
        self.shape = torch.Size([shape])

        # Get mean function and kernel
        M, K_state, K_action, kwargs = get_mean_and_kernel(kernel, mean_type, shape, input_size,
                                                          is_composite=True,
                                                           matern_nu=matern_nu)
        # Default priors work if targets standardized and features min-max normalized
        lengthscale_prior_state = None
        lengthscale_prior_action = None
        outputscale_prior = None

        # Construct mean module
        self.mean_module = M

        # Construct state kernel
        self.state_base_kernel = K_state(batch_shape=self.shape,
                                         active_dims=state_dims,
                                         ard_num_dims=state_ard_num_dims,
                                         **kwargs)
        # Construct action kernel
        self.action_base_kernel = K_action(batch_shape=self.shape,
                                           active_dims=action_dims,
                                           ard_num_dims=action_ard_num_dims,
                                           **kwargs)

        # Construct composite kernel
        self.composite_kernel = self.state_base_kernel * self.action_base_kernel
        self.covar_module = ScaleKernel(self.composite_kernel,
                                        batch_shape=self.shape,
                                        output_device=output_device)

        if use_priors:
            self.covar_module.state_base_kernel.lengthscale = lengthscale_prior_state.mean
            self.covar_module.action_base_kernel.lengthscale = lengthscale_prior_action.mean
            self.covar_module.outputscale = outputscale_prior.mean

    def forward(self, x):
        """Forward pass method for making predictions through the model.  The
        mean and covariance are each computed to produce a MV distribution.

        Parameters:
            x (torch.tensor): The tensor for which we predict a mean and
                covariance used the BatchedGP model.

        Returns:
            mv_normal (gpytorch.distributions.MultivariateNormal): A Multivariate
                Normal distribution with parameters for mean and covariance computed
                at x.
        """
        # Compute mean and covariance in batched form
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class BatchIndependentMultitaskGPModel(ExactGP):
    """GP model for multitask, batched independent models.

     Parameters:
        train_x (torch.tensor): The training features used for Gaussian Process
            Regression.  These features will take shape (B * YD, N, XD), where:

            (i) B is the batch dimension - minibatch size
            (ii) N is the number of data points per GPR - the neighbors considered
            (iii) XD is the dimension of the features (d_state + d_action)

        train_y (torch.tensor): The training labels used for Gaussian Process
            Regression.  These features will take shape (B * YD, N), where:

            (i) B is the batch dimension - minibatch size
            (ii) N is the number of data points per GPR - the neighbors considered

        likelihood (gpytorch.likelihoods.GaussianLikelihood): A likelihood object
            used for training and predicting samples with the BatchedGP model.

        num_tasks (int):  The number of tasks (the dimension of the GPR targets)
            considered for this GPR model.
    """
    def __init__(self, train_x, train_y, likelihood, num_tasks=2):

        self.shape = torch.Size([num_tasks])
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean(batch_shape=self.shape)
        self.covar_module = ScaleKernel(MaternKernel(batch_shape=self.shape),
            batch_shape=self.shape)

    def forward(self, x):
        """Forward pass method for making predictions through the model.  The
        mean and covariance are each computed to produce a MV distribution.

        Parameters:
            x (torch.tensor): The tensor for which we predict a mean and
                covariance used the BatchedGP model.

        Returns:
            mv_normal (gpytorch.distributions.MultivariateNormal): A Multivariate
                Normal distribution with parameters for mean and covariance computed
                at x.
        """
        mean_x = self.mean_module(x)  # Compute the mean at x
        covar_x = self.covar_module(x)  # Compute the covariance at x
        return MultitaskMultivariateNormal.from_batch_mvn(
            MultivariateNormal(mean_x, covar_x))


def get_mean_and_kernel(kernel, mean_type, shape, input_size,
                        is_composite=False, matern_nu=2.5):
    """Utility function to extract mean and kernel for GPR model.

    Parameters:
        kernel (str):  Type of kernel to use for optimization. Defaults to "
            "Matern kernel ('matern'). Other options include RBF
            (Radial Basis Function)/SE (Squared Exponential) ('rbf'), and RQ
            (Rational Quadratic) ('rq').
        mean_type (str): Type of mean function to use for Gaussian Process.
            Defaults to zero mean ('zero').  Other options: linear
            ('linear'), and constant ('constant').")
        shape (int):  The batch shape used for creating this BatchedGP model.
            This corresponds to the number of samples we wish to interpolate.
        input_size (int):  If using a linear mean (else not applicable), this
            is the number of X dimensions.
        is_composite (bool): Whether we are constructing means and kernels for
            a composite GPR kernel.  If True, returns two kernel objects (whose
            attributes are then later set).  Defaults to False.
        matern_nu (float): Value in set if {1/2, 3/2, 5/2} that denotes the power
            to raise the matern kernel evaluation to.  Smaller values allow for
            greater discontinuity.  Only relevant if kernel is matern.  Defaults to
            2.5.
    Returns:
        M (gpytorch.means.Mean): Mean function object for the GPyTorch model.
        K (gptorch.kernels.Kernel): Kernel function object for the GPyTorch model.
    """
    # Create tensor for size
    batch_shape = torch.Size([shape])
    # Determine mean type
    if mean_type == "zero":
        M = ZeroMean(batch_shape=batch_shape)
    elif mean_type == "constant":
        M = ConstantMean(batch_shape=batch_shape)
    elif mean_type == "linear":
        M = LinearMean(input_size, batch_shape=batch_shape)
    else:
        raise Exception("Please select a valid mean type for the GPR. "
                        "Choices are {'zero', 'constant', 'linear'}.")

    # Determine kernel type
    if kernel == "matern":
        K = MaternKernel
    elif kernel == "rbf":
        K = RBFKernel
    elif kernel == "rbf_grad":
        K = RBFKernelGrad
    elif kernel == "rq":
        K = RQKernel
    else:
        raise Exception("Please select a valid kernel for the GPR. "
                        "Choices are {'matern', 'rbf', 'rq'}.")

    # Determine what extra parameters to return
    kwargs = {}
    if kernel == "matern":
        kwargs["nu"] = matern_nu

    # Return means and kernels
    if is_composite:
        return M, K, K, kwargs
    else:
        return M, K, kwargs


def get_priors(heuristic_lengthscales, use_priors=True, is_composite=False):
    """Utility function used to get priors for GPR kernel hypeparameters.

    Parameters:
        use_priors (bool):  Whether to use prior distribution over lengthscale and
            outputscale hyperparameters.  Defaults to False.
        is_composite (bool): Whether we are constructing means and kernels for
            a composite GPR kernel.  If True, returns two sets of priors for
            for each of the two kernels.

    Returns:
        lengthscale_prior (GammaPrior): A Gamma prior distribution on the
            kernel lengthscale hyperparameter.
        outputscale_prior (GammaPrior): A Gamma prior distribution on the
            kernel outputscale hyperparameter.
    """
    # Determine if we use priors
    lengthscale_prior = None
    outputscale_prior = None
    if use_priors:
        lengthscale_prior = GammaPrior(0.01 * heuristic_lengthscales,
                                       0.01 * torch.ones(heuristic_lengthscales.size()))
        print("LENGTHSCALE MEAN: \n{}".format(lengthscale_prior.mean))
        print("LENGTHSCALE VARIANCE: \n{}".format(lengthscale_prior.variance))

    if is_composite:
        return lengthscale_prior, lengthscale_prior, outputscale_prior

    else:
        return lengthscale_prior, outputscale_prior