"""Quick exercise to build intuition for students on the topic of for loops."""

# Problem 1:
print("Problem 1!")
print("_____________________")
x = 5
print("x = 5")
for i in range(0, 5):
    x += 1
    print("For loop", i, "after x += 1:")
    guess = input("What is x now? ")
    print("x is now", x)
# What is x?

print("_____________________")
print("Problem 2!")
print("_____________________")
# Problem 2
x = 1
print("x = 1")
for i in range(1, 5):
    x *= i
    print("i =", i)
    print("For loop", i, "after x *= i:")
    guess = input("What is x now? ")
    print("x is now", x)
# What is x?

print("_____________________")
print("Problem 3!")
print("_____________________")
# Problem 3
x = 1
y = 2
z = 3
print("x = 1, y = 2, z = 3")
for i in range(2):
    x *= 2
    y *= 3
    z *= 4
    print("For loop", i, "after x *= 2:")
    guess_x = input("What is x now? ")
    print("x is now", x)
    print("For loop", i, "after y *= 3:")
    guess_y = input("What is y now? ")
    print("y is now", y)
    print("For loop", i, "after z *= 4:")
    guess_z = input("What is z now? ")
    print("z is now", z)

# What is x + y + z
guess = input("What is x+y+z? ")
print("x+y+z is", x + y + z)

# YOU CAN TYPE YOUR CODE HERE :)
total = 0
for i in range(1, 101):
    if i % 2 == 1:  # If the number is odd
        total += i  # total = total + i
    else:  # If the number is even
        continue
print("Your total is:", total)
# Initialize
tot = 0
for i in range(1, 1000001):
    if i % 3 == 0:
        tot += i  # tot = tot + i
    else:
        continue
print("Sum is:", tot)
