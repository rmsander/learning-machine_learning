"""Tester script for GPyTorch using analytic sine functions."""

import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics import mean_squared_error as mse


def main():
    """Main tester function."""
    # Set parameters
    B = 256  # Number of batches
    N = 100  # Number of data points in each batch
    D = 3  # Dimension of X and Y data
    Ds = 1  # Dimensions for first factored kernel - only needed if factored kernel is used
    EPOCHS = 50  # Number of iterations to perform optimization
    THR = -1e5  # Training threshold (minimum)
    USE_CUDA = torch.cuda.is_available()  # Hardware acceleraton
    MEAN = 0  # Mean of data generated
    SCALE = 1  # Variance of data generated
    COMPOSITE_KERNEL = False  # Use a factored kernel
    USE_ARD = True  # Use Automatic Relevance Determination in kernel
    LR = 0.5  # Learning rate

    # Create training data and labels
    train_x_np = np.random.normal(loc=MEAN, scale=SCALE, size=(B, N, D))  # Randomly-generated data
    train_y_np = np.sin(train_x_np)  # Analytic sine function

    # GPyTorch training
    start = time.time()
    model, likelihood = train_gp_batched_scalar(train_x_np, train_y_np,
                                                use_cuda=USE_CUDA,
                                                composite_kernel=COMPOSITE_KERNEL,
                                                epochs=EPOCHS, lr=LR, thr=THR, ds=Ds,
                                                use_ard=USE_ARD)
    end = time.time()
    print("TRAINING TIME: {}".format(end - start))
    model.eval()
    likelihood.eval()

    # Now evaluate
    test_x_np = np.random.normal(loc=MEAN, scale=SCALE, size=(B, 1, D))
    train_x = preprocess_eval_inputs(train_x_np, train_y_np.shape[-1])
    test_x = preprocess_eval_inputs(test_x_np, train_y_np.shape[-1])

    # Put on GPU
    if USE_CUDA:
        test_x = test_x.cuda()
        train_x = train_x.cuda()

    # Run Inference
    with torch.no_grad():
        f_preds = model(test_x)
        observed_pred = likelihood(model(test_x))
        mean = observed_pred.mean
        variance = observed_pred.variance

    # Generate plots over different dimensions
    for DIM, c in zip([0, 1, 2], ["r", "g", "b"]):
        # Compute mean and unstack
        output = mean.cpu().detach().numpy()
        out_y = np.squeeze(np.array([output[i::B] for i in range(B)]))

        # Reformat, get analytic y, and plot
        x_plot = np.squeeze(test_x_np)
        y_plot_gt = np.sin(x_plot)
        plt.scatter(x_plot[:, DIM], out_y[:, DIM])
        plt.title("Test X vs. Predicted Y, Dimension {}".format(DIM))
        plt.show()

        # Creating color map
        my_cmap_analytic = plt.get_cmap('hot')
        my_cmap_predicted = plt.get_cmap('cool')

        # Plot ground truth
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x_plot[..., 0], x_plot[..., 1], y_plot_gt[..., DIM],
                        antialiased=True, cmap=my_cmap_analytic)
        plt.title("Analytic, Ground Truth Surface, Dimension {}".format(DIM))
        plt.show()

        # Plot the predicted values
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x_plot[..., 0], x_plot[..., 1], out_y[..., DIM],
                        antialiased=True, cmap=my_cmap_predicted)
        plt.title("Predicted Surface, Dimension {}".format(DIM))
        plt.show()

    # Compute average RMSE
    y_true = np.squeeze(np.sin(test_x_np))
    y_pred = out_y
    rmse = np.sqrt(mse(y_true, y_pred))
    print("RMSE: {}".format(rmse))
    return rmse

def train_gp_batched_scalar(Zs, Ys, use_cuda=False, epochs=10,
                            lr=0.1, thr=0, use_ard=False, composite_kernel=False,
                            ds=None, global_hyperparams=False,
                            model_hyperparams=None):
    """Computes a Gaussian Process object using GPyTorch. Each outcome is
    modeled as a single scalar outcome.
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
        use_ard (bool):  Whether to use Automatic Relevance Determination (ARD)
            for the lengthscale parameter, i.e. a weighting for each input dimension.
            Defaults to False.
        composite_kernel (bool):  Whether to use a composite kernel that computes
            the product between states and actions to compute the variance of y.
        ds (int): If using a composite kernel, ds specifies the dimensionality of
            the state.  Only applicable if composite_kernel is True.
        global_hyperparams (bool):  Whether to use a single set of hyperparameters
            over an entire model.  Defaults to False.
        model_hyperparams (dict):  A dictionary of hyperparameters to use for
            initializing a model.  Defaults to None.
    Returns:
        model (BatchedGP): A GPR model of BatchedGP type with which to generate
            synthetic predictions of rewards and next states.
        likelihood (GaussianLikelihood): A likelihood object used for training
            and predicting samples with the BatchedGP model.
    """
    # Preprocess batch data
    B, N, XD = Zs.shape
    YD = Ys.shape[-1]
    batch_shape = B * YD

    if use_cuda:  # If GPU available
        output_device = torch.device('cuda:0')  # GPU

    # Format the training features - tile and reshape
    train_x = torch.tensor(Zs, device=output_device)
    train_x = train_x.repeat((YD, 1, 1))

    # Format the training labels - reshape
    train_y = torch.vstack(
        [torch.tensor(Ys, device=output_device)[..., i] for i in range(YD)])

    # initialize likelihood and model
    likelihood = GaussianLikelihood(batch_shape=torch.Size([batch_shape]))

    # Determine which type of kernel to use
    if composite_kernel:
        model = CompositeBatchedGP(train_x, train_y, likelihood, batch_shape,
                          output_device, use_ard=use_ard, ds=ds)
    else:
        model = BatchedGP(train_x, train_y, likelihood, batch_shape,
                          output_device, use_ard=use_ard)

    # Initialize the model with hyperparameters
    if model_hyperparams is not None:
        model.initialize(**model_hyperparams)

    # Determine if we need to optimize hyperparameters
    if global_hyperparams:
        if use_cuda:  # Send everything to GPU for training
            model = model.cuda().eval()

            # Empty the cache from GPU
            torch.cuda.empty_cache()
            gc.collect()  # NOTE: Critical to avoid GPU leak
            del train_x, train_y, Zs, Ys, likelihood

        return model, model.likelihood

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # "Loss" for GPs - the marginal log likelihood
    mll = ExactMarginalLogLikelihood(likelihood, model)

    if use_cuda:  # Send everything to GPU for training
        model = model.cuda()
        likelihood = likelihood.cuda()
        train_x = train_x.cuda()
        train_y = train_y.cuda()
        mll = mll.cuda()

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

    # Run the optimization loop
    for i in range(epochs):
        loss_i = epoch_train(i)
        if i % 10 == 0:
            print("LOSS EPOCH {}: {}".format(i, loss_i))
        if loss_i < thr:  # If we reach a certain loss threshold, stop training
            break

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

if __name__ == '__main__':
    all_rmses = []
    for i in range(10):
        all_rmses.append(main())
    print("AVERAGED RMSE: {}".format(np.mean(all_rmses)))
