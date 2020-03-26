"""Quick, simple exercise for showing different Python operations on lists."""

# Exercise 1: List only exercises
my_list = []
print("Before append:", my_list)
my_list.append(5)
print("After first append:", my_list)
my_list.append(10)
print("After second append:", my_list)

my_list.pop(0)
print("After pop:", my_list)

my_list.remove(10)
print("After remove:", my_list)

# Exercise 2: List and String exercises
my_list = [1, 2, 5, 6, 8]
my_list.append(11)
print(my_list)
"""prints -> [1,2,5,6,8,11]"""

my_list.pop(2)
print(my_list)
"""prints -> [1,2,6,8,11]"""

my_list.remove(8)
print(my_list)
"""prints -> [1,2,6,11]"""

# Let's store numbers 1-10000!
numbers = []
for i in range(1, 10001):
    numbers.append(i)

# Creates a 32-bit number
string = ""
for i in range(32):
    string += 0 or 1
