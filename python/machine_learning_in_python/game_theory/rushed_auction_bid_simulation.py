"""Script to run dutch auction simulation with variation in the parameters of
the quasi-hyperbolic bidder.

Namely, this captures a Dutch Auction system with the following parameters:

        1. alpha: The exponential discounting factor.
        2. beta: The quasi-hyperbolic discounting factor.
        3. gamma: The time pressure factor. Causes the player to
            over/under-estimate the probability of losing a bid in a Dutch
            Auction.

By default, the starting price is 10, and each day the price drops by 1.
"""
import xlrd
from xlrd import open_workbook
from xlutils.copy import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Specify duration of the bid
NUM_DAYS = 40


def make_alpha_beta_bids_dict():
    """Function to loop over and create dictionary mapping
    (alpha, beta) pairs to days bids are placed."""
    # Initialize dictionary
    alpha_beta_gamma_bids_dict = {}

    # Specify if we should keep variables constant
    print("For reach, please enter y (yes) or n (no).")
    alpha_constant = input("Keep alpha constant? --> ")
    beta_constant = input("Keep beta constant? --> ")
    gamma_constant = input("Keep gamma constant? --> ")

    # Specify granularity for parameter sweeps
    if alpha_constant == "n":
        alpha_step_size = float(
            input("Please specify a step size for alpha (e.g. 0.1): --> "))
    elif alpha_constant == "y":
        print("Since alpha constant, step size is not needed.")

    if beta_constant == "n":
        beta_step_size = float(
            input("Please specify a step size for beta (e.g. 0.1): --> "))
    elif beta_constant == "y":
        print("Since beta constant, step size is not needed.")

    if gamma_constant == "n":
        gamma_step_size = float(
            input("Please specify a step size for gamma (e.g. 0.1): --> "))
    elif gamma_constant == "y":
        print("Since gamma constant, step size is not needed.")

    # Specify ranges for alpha, beta, and gamma parameters
    if alpha_constant == "n":
        alpha_range = eval(
            input("Please provide a range for alpha in the form of [low, high] --> "))
        alpha_low, alpha_high = alpha_range[0], alpha_range[1]
        alphas = np.arange(alpha_low, alpha_high + alpha_step_size, alpha_step_size)
    elif alpha_constant == "y":
        alphas = np.array(
            float(input("Please provide a constant value for alpha --> "))).reshape(1)

    if beta_constant == "n":
        beta_range = eval(
            input("Please provide a range for beta in the form of [low, high] --> "))
        beta_low, beta_high = beta_range[0], beta_range[1]
        betas = np.arange(beta_low, beta_high + beta_step_size, beta_step_size)
    elif beta_constant == "y":
        betas = np.array(
            float(input("Please provide a constant value for beta --> "))).reshape(1)

    if gamma_constant == "n":
        gamma_range = eval(
            input("Please provide a range for gamma in the form of [low, high] --> "))
        gamma_low, gamma_high = gamma_range[0], gamma_range[1]
        gammas = np.arange(gamma_low, gamma_high + gamma_step_size, gamma_step_size)
    elif gamma_constant == "y":
        gammas = np.array(
            float(input("Please provide a constant value for gamma --> "))).reshape(1)

    # Use granularity to get number in each parameter
    num_alphas = alphas.size
    num_betas = betas.size
    num_gammas = gammas.size
    print("Dimensions of output array:")
    print("Num Alphas: {}, Range: {}".format(num_alphas, alphas))
    print("Num Betas: {}, Range: {}".format(num_betas, betas))
    print("Num Gammas: {}".format(num_gammas, gammas))

    # Initialize 3D array and ranges of variables to loop over
    alpha_beta_gamma_array = np.zeros(
        (num_alphas, num_betas, num_gammas)).astype(np.uint32)

    # Get total number of parameter combinations
    N = num_alphas * num_betas * num_gammas

    # Make an array for points
    point_array = np.zeros((N, 3))
    days_array = np.zeros(N).astype(np.uint32)

    # Loop over alpha, beta, and gamma
    idx = 0
    for a, alpha in enumerate(alphas):  # Loop over "rows"
        for b, beta in enumerate(betas):  # Loop over "columns"
            for g, gamma in enumerate(gammas):  # Loop over "page"

                # Get the bid day for the specific parameter configuration
                bid_day = loop_helper_analytic(alpha, beta, gamma)

                # Add to dictionary and array
                alpha_beta_gamma_bids_dict[(alpha, beta, gamma)] = bid_day
                alpha_beta_gamma_array[a, b, g] = bid_day

                # Add the values to the points arrays
                point_array[idx, :] = np.array([alpha, beta, gamma])
                days_array[idx] = bid_day
                idx += 1

    # Add visualization code
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Separate plotted data
    x_plot = point_array[:, 0]  # Alpha - first column
    y_plot = point_array[:, 2]  # Gamma - second column
    z_plot = point_array[:, 1]  # Beta - third column

    # Create a 3D scatter plot with heatmap
    ax.scatter(x_plot, y_plot, z_plot, c=days_array,
               marker="o", label="Bid Day Heat Map")
    ax.set_xlabel("Alpha")
    ax.set_ylabel("Gamma")
    ax.set_zlabel("Beta")
    plt.title("Bidding Days vs. Alpha, Beta, and Gamma")
    plt.savefig("bid_surface.png")
    plt.show()

    # Store 3D array as | alpha | beta | gamma | bid day|
    concantenated_data = np.hstack((point_array, days_array.reshape((-1, 1))))
    df = pd.DataFrame(concantenated_data)

    # Specify the column headers
    df.columns = ["Alpha", "Beta", "Gamma", "Bid Day"]

    # And output to excel
    df.to_excel("bid_days.xlsx")

    # Display and return
    print("BID DICTIONARY: \n{}".format(alpha_beta_gamma_bids_dict))
    print("BID ARRAY: \n{}".format(np.squeeze(alpha_beta_gamma_array)))
    return alpha_beta_gamma_bids_dict, alpha_beta_gamma_array


def loop_helper_analytic(alpha, beta, gamma):
    """Helper function to compute forward utilities and determine when to
    bid using analytic expressions for utility.

    Parameters:
        alpha (float): The exponential discounting factor.
        beta (float): The quasi-hyperbolic discounting factor.
        gamma (float): The time pressure factor. Causes the player to
            over/under-estimate the probability of losing a bid in a Dutch
            Auction.
    """
    # Now we want to loop over values to find day bid is placed
    bid_placed = False  # Exit condition
    current_day = 1  # Used to increment
    while not bid_placed:  # Continue looping until bid placed

        # Initialize values list
        values = []

        # Get column data (utilities into the future) for current_day
        for day_forward in range(current_day, NUM_DAYS+1):

            # Calculate probability of not losing (weighted by gamma)
            prob_not_losing = np.prod(
                [1-(1/(NUM_DAYS+1-day) * gamma) for day in range(current_day, day_forward + 1)])

            # Calculate discounted utility with quasi-hyperbolic preferences
            discount_factor = alpha ** (abs(day_forward-current_day))
            if day_forward > current_day:  # Only factor in beta if next day
                discount_factor *= beta
            discounted_value = day_forward * discount_factor

            # Discount return with probability of not losing (expected utility)
            values.append(prob_not_losing * discounted_value)

        # Check if max is first element
        if np.argmax(values) == 0:
            bid_placed = True  # Exit condition
            day_of_bid = current_day  # Variable we return

        # Increment the current day
        current_day += 1

    return day_of_bid


def main():

    # Call the function to make the dictionary
    bids_dict, bids_array = make_alpha_beta_bids_dict()


if __name__ == '__main__':
    main()
