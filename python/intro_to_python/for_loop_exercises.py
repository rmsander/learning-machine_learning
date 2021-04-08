"""Exercise for showing how loops/iteration can be used effectively in Python to
solve problems that are too complicated/take too long to solve by hand."""

# We'll use this to do mathematical operations
import math

# These exercises will show how we can use iteration to solve problems!
print("These exercises will show how we can use iteration to solve problems!")

# Exercise 1: Finding Sums of Sequences
repeat = "yes"
print("Exercise 1")
print("_____________________________________")
print("This program allows us to find sums of lots of numbers VERY quickly.")
print("_____________________________________")
while repeat == "yes":
    upper = int(input("Type a REALLY BIG number: "))
    print("Ok, let's find the sum from 1 to", upper, ". Ready? Press enter.")
    ready = input()
    tot = 0
    for i in range(1, upper + 1):
        tot += i
    print("The sum of numbers from 1 to", upper, "is", tot)
    print("_____________________________________")
    print("Now let's find the sum of ODD numbers from 1 to", upper,
          ". Ready? Press enter.")
    ready = input()
    tot2 = 0
    for i in range(1, upper + 1):
        if i % 2 == 1:
            tot2 += i
    print("The sum of ODD numbers from 1 to", upper, "is", tot2)
    print("_____________________________________")
    print("Now let's find the sum of EVEN numbers from 1 to", upper,
          ". Ready? Press enter.")
    ready = input()
    tot3 = 0
    for i in range(1, upper + 1):
        if i % 2 == 0:
            tot3 += i
    print("The sum of EVEN numbers from 1 to", upper, "is", tot3)
    print("_____________________________________")
    repeat = input("Would you like to do this again?  Please enter yes or no. ")
    print("_____________________________________")

# Exercise 2
repeat2 = "yes"
print("Exercise 2: Binary Number Counter!")
print("_____________________________________")
print(
    "This program will let us figure out what number a binary number "
    "represents using for loops!")
print("_____________________________________")
while repeat2 == "yes":
    x = input(
        "What is your binary number? Please enter as a sequence of 1's and "
        "0's (example: 1001): ")
    tot4 = 0
    x_list = list(x)
    x_list.reverse()
    for i in range(len(x_list)):
        if int(x_list[i]) == 1:
            tot4 += 2 ** i
    print("This binary number sequence represents: ", tot4)
    repeat2 = input(
        "Would you like to do this again?  Please enter yes or no. ")

# Exercise 3
repeat3 = "yes"
print("Exercise 3: Binary Number Maker!")
print("_____________________________________")
print("This program will convert a number into a binary number! ")
print("_____________________________________")
while repeat3 == "yes":
    num = []
    x = int(input("What is your number? "))
    remainder = x
    while remainder >= 1:
        if remainder % 2 == 1:
            num.append(1)
        else:
            num.append(0)
        remainder = math.floor(remainder / 2)
    num.reverse()
    string = ""
    for i in range(len(num)):
        string = string + str(num[i])
    print("In binary, the number", x, "is", string)
    repeat3 = input(
        "Would you like to do this again?  Please enter yes or no. ")


# Exercise 4 - note that this uses a function
def perfect_square(x):
    arr = []
    for i in range(x):
        if i ** .5 % 1 == 0:
            arr.append(i)
    return arr


print(perfect_square(100))
