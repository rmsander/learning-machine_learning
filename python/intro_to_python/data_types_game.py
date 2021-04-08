"""A game-style exercise where students can practice identifying different
data types in Python."""

# Name That Data Type!
score = 0
print("Name That Data Type!")
print("For each answer, please respond with one of:")
print(" ")
print("int, float, str, bool, NoneType")
print("____________________________")
ans_1 = input("What kind of data type is: 6.0? ")
if ans_1 == "float":
    score += 1
    print("That's right!")
else:
    print("Nice try! 6.0 is a float.")
ans_2 = input("What kind of data type is: 7? ")
if ans_2 == "int":
    score += 1
    print("That's right!")
else:
    print("Nice try! 7 is a int.")
ans_3 = input("What kind of data type is: None? ")
if ans_3 == "NoneType":
    score += 1
    print("That's right!")
else:
    print("Nice try! None is a NoneType.")
ans_4 = input("What kind of data type is: word? ")
if ans_4 == "str":
    score += 1
    print("That's right!")
else:
    print("Nice try! Python is a str.")
ans_5 = input("What kind of data type is: False? ")
if ans_5 == "bool":
    score += 1
    print("That's right!")
else:
    print("Nice try! False is a bool.")
ans_6 = input("What kind of data type is: True? ")
if ans_6 == "bool":
    score += 1
    print("That's right!")
else:
    print("Nice try! False is a bool.")
ans_7 = input("What kind of data type is: 33.333333? ")
if ans_7 == "float":
    score += 1
    print("That's right!")
else:
    print("Nice try! 33.333333 is a float.")
ans_8 = input("What kind of data type is: 1==0? ")
if ans_8 == "bool":
    score += 1
    print("That's right!")
else:
    print("Nice try! 1==0 is a bool.")
ans_9 = input("What kind of data type is: INJAZ? ")
if ans_9 == "str":
    score += 1
    print("That's right!")
else:
    print("Nice try! INJAZ is a str.")
ans_10 = input("What kind of data type is: -43? ")
if ans_10 == "int":
    score += 1
    print("That's right!")
else:
    print("Nice try! -43 is a int.")
print("____________________________________________")
print("Nice job! Your final score is: ", score, "/10.")

print("Challenge Problem!")
ans_11 = input("What kind of data type is: str(int(float(-100.005)))? ")
if ans_11 == "str":
    print("That's right!")
else:
    print("Nice try!  The correct answer is str.")
