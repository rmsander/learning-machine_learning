import xlrd
from xlrd import open_workbook
from xlutils.copy import copy
import numpy as np

def loop_helper_excel(alpha, beta, sheet_path):
    rb = open_workbook(sheet_path)
    wb = copy(rb)
    sheet = wb.get_sheet(0)

    # Write alpha/beta values
    sheet.write(1, 1, beta)
    sheet.write(2, 1, alpha)

    # Save sheet
    wb.save(sheet_path)
    print("alpha: {}, beta: {}".format(alpha, beta))

    # Reopen file
    book = xlrd.open_workbook(sheet_path)


def loop_helper_analytic(alpha, beta):
    """Helper function to compute forward utilities and determine when to
    bid using analytic expressions for utility."""
    # Now we want to loop over values to find day bid is placed
    bid_placed = False  # Exit condition
    current_day = 1  # Used to increment
    while not bid_placed:  # Continue looping until bid placed

        # Initialize values list
        values = []

        # Get column data (utilities into the future) for current_day
        for day_forward in range(current_day, 11):

            # Calculate probability of not losing
            prob_not_losing = np.prod(
                [1-(1/(11-day)) for day in range(current_day, day_forward+1)])

            # Calculate discounted utility with quasi-hyperbolic preferences
            discount_factor = beta * (alpha ** (abs(day_forward-current_day)))
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

def make_alpha_beta_bids_dict():
    """Function to loop over and create dictionary mapping
    (alpha, beta) pairs to days bids are placed."""
    # Initialize dictionary
    alpha_beta_bids_dict = {}

    # Initialize 2D array and ranges of variables to loop over
    alpha_beta_array = np.zeros((10, 10)).astype(np.uint32)
    alpha_range = np.arange(0.1, 1.1, step=0.1)  # Note: Doesn't take last number
    beta_range = np.arange(0.1, 1.1, step=0.1)   # Note: Doesn't take last number

    # Loop over alpha and beta
    for a, alpha in enumerate(alpha_range):
        for b, beta in enumerate(beta_range):

            # Get the bid day
            bid_day = loop_helper_analytic(alpha, beta)

            # Add to dictionary and array
            alpha_beta_bids_dict[(beta, alpha)] = bid_day
            alpha_beta_array[a, b] = bid_day

    # Format alpha_beta_array to show row and column headers
    # Add column of headers for alpha values
    alpha_headers = np.array(
        ["Alpha (col A)/Beta (row 1)"] + ["{}".format(round(a, 1)) for a in alpha_range]).reshape((-1, 1))

    # Add row of headers for beta values
    beta_headers = np.array(
        ["{}".format(round(b, 1)) for b in beta_range])

    # Concatenate arrays
    # Add column headers
    alpha_beta_array = np.vstack((beta_headers, alpha_beta_array))

    # Add row headers
    alpha_beta_array = np.hstack((alpha_headers, alpha_beta_array))

    import pandas as pd
    df = pd.DataFrame(alpha_beta_array)
    df.to_excel("bid_days.xlsx")

    # Display and return
    print("BID DICTIONARY: \n{}".format(alpha_beta_bids_dict))
    print("BID ARRAY: \n{}".format(alpha_beta_array))
    return alpha_beta_bids_dict, alpha_beta_array


def main():

    # Call the function to make the dictionary
    bids_dict, bids_array = make_alpha_beta_bids_dict()


if __name__ == '__main__':
    main()
