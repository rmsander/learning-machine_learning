"""An in-class exercise which will help students to build intuition for
probability using Python.  Concepts are nuanced (Law of Large Numbers),
but this exercise seeks to circumvent this nuance by intuitively showing
students the trade-off between computational complexity and convergence of a
sequence of random variables to its mean.

Scores
"""

# Python has many packages for us to use - this is how we use them!
import scipy.stats
import time

# Write a function to find mean!
# Step 1: Define function and input(s)
def expectation(X):  # X is a list of numbers
    # Step 2: Write body of code
    n = len(X)
    tot = 0
    for i in range(n):
        tot += X[i]
    # Step 3: Return y-value
    return tot / n


# Now let's find variance
# Step 1: Define function and input(s)
def std(X):
    # Step 2: Write body of code
    n = len(X)
    X2 = []
    for i in range(n):
        X2.append(X[i] ** 2)
    std = (expectation(X2) - expectation(X) ** 2) ** .5
    # Step 3: Return statement
    return std


# Now let's find the average of coin flips!
# Step 1: Define function
def coin_flips(n):  # n is the number of dice rolls
    # Step 2: Write body of code
    p = 0.5
    flips = scipy.stats.bernoulli.rvs(.5, size=n)
    heads = 0
    tails = 0
    for i in range(n):
        if flips[i] == 1:
            heads += 1
        else:
            tails += 1
    print("Number of heads:", heads, "Number of tails:", tails)
    print("Mean of coin flips: ", expectation(flips))
    print("Standard Deviation of coin flips: ", std(flips))
    return expectation(flips)


# Function for scoring final exercise
def compute_points(mu, time_to_compute, target=0.5):
    return ((1 - abs(target - mu) - ((time_to_compute) / 100)) ** 10) * 100

################################################################################
n = 100000  # YOUR CODE HERE - THIS IS THE NUMBER OF FLIPS YOU HAVE
################################################################################
print("Your number of coin flips is", n)

# Compute the mean of all coin flips!
time_start = time.time()
mean = coin_flips(n)
time_end = time.time()

# Now score how well you did!
points = compute_points(mean, time_end-time_start)
print("Your points are:", points)
