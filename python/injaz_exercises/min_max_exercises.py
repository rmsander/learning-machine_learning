"""Functions for showing students how we can use min/max operations to find
certain largest/smallest/second largest/smallest numbers in lists and
sequences."""


# finds the second largest number in a sequence
def second_largest_number(A):
    maximum = min(A)
    A.remove(maximum)
    return max(A)


# finds the second smallest number in a sequence
def second_smallest_number(A):
    minimum = max(A)
    A.remove(minimum)
    return min(A)


# v has components in the x and y directions
def find_vector_length(v):
    return (v[0] ** (2) + v[1] ** (2)) ** (1 / 2)
