"""Exercise for helping students practice with dictionary objects using a
cipher they create/experiment with."""

# Here's how we initialize a dictionary!
my_dictionary = {}

# Here's how we add a key-value pair to the dictionary
my_dictionary["key"] = "value"

# Here's how we make 5 a key, and "a" a value
my_dictionary[5] = "a"

# Cipher dictionary!

# Step 1: Initialization
cipher = {}

# Step 2: Map letters to other letters!
cipher["a"] = "b"
cipher["b"] = "c"
cipher["c"] = "d"
# ................
cipher["y"] = "z"
cipher["z"] = "a"

# Example 2: Numbers to Squares!

# Step 1: Initialization
squares = {}

# Step 2: Map numbers to their squares using a for loop!
for i in range(1, 11):
    squares[i] = i ** 2
print(squares)
