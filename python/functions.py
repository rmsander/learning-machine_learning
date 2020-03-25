"""Quick fill-in-the-blank exercises for students to practice with functions
in Python."""

#Define a function first
def my_function(A,B,C):
  return A+B+C

#Now let's care this function!
x = my_function(3,4,5)
#^Here, what will x be?


#Here's how we make functions in Python!

#Step 1: Define the function and arguments!
def my_function(A,B,C):
    #Step 2: Write steps in function
    <Function content>
    #Step 3: Return important
    return something


#Example: y = 3x
#_______________
#Step 1: Define the function and arguments!
def linear(x):
    #Step 2: Write steps in function
    y = 3*x
    #Step 3: Return something important
    return y

#Example: return a sum of a list of values in a dictionary
#_______________
#Step 2: Define the function and arguments
def sum_dictionary(H):
    #Step 2: Write steps in function
    values = list(H.values())
    total = 0
    for i in range(len(values)):
        total += values[i]
    #Step 3: Return something important
    return total




