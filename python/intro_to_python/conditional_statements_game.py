"""A game-style exercise where students can practice with conditional
statements in Python."""

print("Welcome to Conditional Puzzles!")
print("_______________________")
print("For each question, please type in either True or False, and press enter")
print("_______________________")
print("Please make sure that you type these as True or False.")
print("_______________________")
score = 0
print("w = 1, x = 5, y = 4, z = 2")
ans_1 = input("What is: x > y and y > z? ")
if ans_1 == "True":
    score += 1
    print("That's correct!")
else:
    print("Sorry, that's not correct.  The correct answer is True.")

ans_2 = input("What is: x > y and y <= z? ")
if ans_2 == "False":
    score += 1
    print("That's correct!")
else:
    print("Sorry, that's not correct.  The correct answer is False.")

ans_3 = input("What is: z == x or z != x? ")
if ans_3 == "True":
    score += 1
    print("That's correct!")
else:
    print("Sorry, that's not correct.  The correct answer is True.")

ans_4 = input("What is: z > 4 or z >= 2? ")
if ans_4 == "True":
    score += 1
    print("That's correct!")
else:
    print("Sorry, that's not correct.  The correct answer is True.")

ans_5 = input("What is: (x > y and y > 100) or z < 1? ")
if ans_5 == "False":
    score += 1
    print("That's correct!")
else:
    print("Sorry, that's not correct.  The correct answer is False.")

ans_6 = input("What is: (x > y and y > z) and z > x? ")
if ans_6 == "False":
    score += 1
    print("That's correct!")
else:
    print("Sorry, that's not correct.  The correct answer is False.")

ans_7 = input("What is: (x > y or y > z) and (x == 5 and y == 1) ? ")
if ans_7 == "False":
    score += 1
    print("That's correct!")
else:
    print("Sorry, that's not correct.  The correct answer is False.")

ans_8 = input("What is: x > 1 and y < 5? ")
if ans_8 == "True":
    score += 1
    print("That's correct!")
else:
    print("Sorry, that's not correct.  The correct answer is True.")
print("_____________________________________________________")
print("Challenge Problem!")
ans_challenge = input(
    "What is: (((x != y or x == y) and (w > x or w <= x)) and ((y == z or y "
    "!= z) and (z == w or z != w)))? ")
if ans_challenge == "True":
    score += 1
    print("That's correct!")
else:
    print("Sorry, that's not correct.  The correct answer is True.")
print("_____________________________________________________")
print("About this challenge problem:")
print(
    "This problem captures what's known in set algebra as complements.  For "
    "any x and y, it must be True that x != y or x == y.  This fact becomes "
    "very useful in probability.")

print("Nice job!  Your score is ", score, "/8")
