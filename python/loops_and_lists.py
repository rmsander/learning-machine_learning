"""Quick, simple exercise for showing different Python operations on lists
using for loops."""


# Nested list exercise!
# Let's make a nested for loop to find the sum of all odd numbers in these
# lists of lists

# Step 1: Define a function!
def numbers(list_of_lists):
    # Step 2: Function body!
    total = 0
    # Loop through the list of lists
    for i in range(len(list_of_lists)):
        # Loop through each list
        for j in range(len(list_of_lists[i])):
            # Check if a number is odd
            if list_of_lists[i][j] % 2 == 1:
                # Add to total
                total += list_of_lists[i][j]
    # Step 3: Return total!
    return total


# Let's define a function that, given a list of numbers, returns a product of
# all the odd numbers in that list

# Step 1: Define the function!
def injaz(A):  # A is our list
    # Step 2: Body of the function
    total = 1  # For keeping track of the product
    for i in range(0, len(A)):
        if A[i] % 2 == 1:  # If the number is odd (1,3,5,7,...)
            total *= A[i]  # Same as total = total * A[i]
    # Step 3: Return our y-value!
    return total  # Returns the product


# Now let's use our function!
my_list = [i for i in range(100)]  # Fancy way of making a list
product = injaz(my_list)
print(product)


# Let's make a function that finds the sum of all even numbers in all lists
# from a list of lists
# Step 1: Define function!
def example(list_of_lists):
    # Step 2: Find total!
    total = 0
    for list_ in list_of_lists:
        for i in range(len(list_)):
            if list_[i] % 2 == 0:
                total += list_[i]
    # Step 3: Return total!
    return total


# Let's use our function now!
A = [[i for i in range(j)] for j in range(10)]
summ = example(A)
print(A)
print(summ)

B = [[i for i in range(j)] for j in range(100)]
summ = example(B)
print(B)
print(summ)

# Find a sum of a list of lists
list_of_lists = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
total += 1
for list_ in lists_of_lists:
    for i in range(len(list_)):
        total += list_


# First exercise!
def my_function(my_list):
    # Find the sum of odd numbers and the product of even numbers!
    sum_odd = 0
    product_even = 1
    for i in range(0, len(my_list)):
        if my_list[i] % 2 == 1:
            sum_odd += my_list[i]
        else:
            product_even *= my_list[i]
    print(my_list)
    y = sum_odd + product_even
    return y
