"""Introductory exercises to show students how different data types work in
Python."""

# This imports a Python package!
import math

# These are ints
x = 20
y = -4

# These are floats
a = 14.135
b = -2.324

# Let's turn a float into an int!
z = int(a)
print("Let's turn a float into an int!")
print(z)

# Let's find the absolute value of a number!
w = abs(y)
print("Let's find the absolute value of a number!")
print(w)

# Let's find the largest number!
i = max(10, 20, 30, 40, 50, 50.01)
print("Let's find the largest number!")
print(i)

# Let's find the smallest number!
j = min(10, 20, 30, 40, 50, 50.01)
print("Let's find the smallest number!")
print(j)

# Let's find the square root of a number!
k = math.sqrt(16)
print("Let's find the square root of a number!")
print(k)


#Let's Initialize A String!
words = "My String"
print(words)

#Let's break this into different words!
split_words = words.split()
print(split_words)

#Let's find the y in our string!
character = words[1]
print(character)

#Here are how we create Booleans
x = (1 == 0)
print(x)
"""prints False"""

y = bool(1)
print(y)
"""prints True"""




