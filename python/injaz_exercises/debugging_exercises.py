"""Exercises for students to complete in which they have to debug them for it to
work properly.  The locations of bugs are given with in-line comments."""

#Exercise 1: Find the product of all numbers from 1-200 divisible by 9!
prod = 1
for i in range(1,200): #First bug here!
  if i % 9 == 1: #Second bug here!
    prod *= i
if prod == 1106883210748590329822115169309534126080000:
  print("Nice job! You found the bugs!")
else:
  print("So close!  Try again.  Look at lines 5 and 6.")


#Exercise 2: Find the Syntax Error!
def message():
  for i in range(10):
    if i == 10:
    print("Leen and Lara are great bug detectives!") #Hint: Check here!
    else:
      continue
  return "Done"

#Exercise 3: Find the largest number in the shortest list!
import math
def largest_shortest(A): #A is a list of lists
  lengths = []
  maxes = []
  for one_list in A:
    lengths.append(len(one_list))
    maxes.append(min(one_list)) #Bug 1 here!
  #Now find the index!
  shortest_length = math.inf
  index = 0
  for i in range(len(A)):
    if lengths[i] > shortest_length:  #Bug 2 here!
      shortest_length = lengths[i]
      index = i
  return maxes[i]

#Test case:
A = [[100-i for i in range(100-j)] for j in range(1,100)]
if largest_shortest(A) == 100:
  print("Great job! You fixed the bugs in this code!")
else:
  print("Nice try!  Try looking at lines 30 and 35")