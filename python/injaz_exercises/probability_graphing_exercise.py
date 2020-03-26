"""Simple exercise to show how Python can be used for probability and
graphing/plotting.  Looks at Law of Large Numbers in an intuitive way."""

# Import these packages
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Initialize output variables!
numbers = []
x = []
y = []
total = 0

# Step 2: What is our range?
for n in range(1, 1000):
    x.append(n)
    # Step 3: Make a list and find
    for i in range(0, n):
        # Now we want to append
        numbers.append(np.random.normal(loc=0.0, scale=100.0, size=None))

    # Step 4: Find the mean!  How does it change as we increase n?
    for i in range(len(numbers)):
        total += numbers[i]

    # Find the mean
    mean = total / n
    y.append(abs(mean))
    mean = 0
    total = 0
    numbers = []


# Now that we have all our numbers, we can graph/plot them!
plt.plot(x, y)
plt.savefig('graph.png')
