"""A game-style exercise where students can practice with math in Python."""

# Welcome to Math Puzzles!
print("Welcome to Math Puzzles!")
print("For each question, please type in the answer, and press enter")

score = 0
print(" ")
print("variables")
print("w = 1, x = 2, y = 5, z = 7")
print("___________________________")
ans_1 = int(input("What is x+y? "))
if ans_1 == 7:
    score += 1
    print("That is correct!  Nice job.")
else:
    print("Sorry, that's incorrect.  The answer was: 7.")

ans_2 = int(input("What is z+w? "))
if ans_2 == 8:
    score += 1
    print("That is correct!  Nice job.")
else:
    print("Sorry, that's incorrect.  The answer was: 8.")

ans_3 = int(input("What is x*y + w? "))
if ans_3 == 11:
    score += 1
    print("That is correct!  Nice job.")
else:
    print("Sorry, that's incorrect.  The answer was: 11.")

ans_4 = int(input("What is z**2? "))
if ans_4 == 49:
    score += 1
    print("That is correct!  Nice job.")
else:
    print("Sorry, that's incorrect.  The answer was: 49.")

ans_5 = float(input("What is w/x? "))
if ans_5 == .5:
    score += 1
    print("That is correct!  Nice job.")
else:
    print("Sorry, that's incorrect.  The answer was: 1/2.")

ans_6 = int(input("What is y%x? "))
if ans_6 == 1:
    score += 1
    print("That is correct!  Nice job.")
else:
    print("Sorry, that's incorrect.  The answer was: 1.")

ans_7 = int(input("What is z*w*x*y?"))
if ans_7 == 70:
    score += 1
    print("That is correct!  Nice job.")
else:
    print("Sorry, that's incorrect.  The answer was: 70.")

ans_8 = int(input("What is x+5? "))
if ans_8 == 7:
    score += 1
    print("That is correct!  Nice job.")
else:
    print("Sorry, that's incorrect.  The answer was: 7.")

ans_9 = int(input("What is x*y/2? "))
if ans_9 == 5:
    score += 1
    print("That is correct!  Nice job.")
else:
    print("Sorry, that's incorrect.  The answer was: 5.")

ans_10 = int(input("What is 3*x + 4*y? "))
if ans_10 == 26:
    score += 1
    print("That is correct!  Nice job.")
else:
    print("Sorry, that's incorrect.  The answer was: 26.")

print("Nice job!  Your score is ", score, "/10")

print("Challange Problem!")
print("w = 5, x = 4, y = 10, z = 5")
ans_11 = int(input("What is ((((w%x)*y)/z)*3*(x**3)+3)%y? "))
if ans_11 == 7:
    score += 1
    print("That is correct!  Nice job.")
else:
    print("Sorry, that's incorrect.  The answer was: 7.")

print("Nice job!  Your score is ", score, "/10")
