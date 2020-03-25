"""A game-style exercise where students can practice with physics in Python
(one of the other subjects taught through this program)."""

# Exercise 1: What is the final velocity?
print("Exercise 1: What is the final velocity of a falling soccer ball?")
print("______________________________________")
v0 = float(input("What is the initial velocity of the ball? "))
a = 9.81
print("On earth, gravity is 9.81 m/s^2.")
t = float(input("How long has the ball been falling? "))
v = v0 + a * t
print("After", t, "seconds of falling, the final velocity of the ball will be:",
      int(v), "m/s.")

print("______________________________________")
# Exercise 2: What is the displacement?
print("Exercise 2: What is the displacement of the car?")
print("______________________________________")
v0 = float(input("What is the initial velocity of the car? "))
v = float(input("What is the final velocity of the car? "))
t = float(input("How long has the car been traveling? "))
dx = (v + v0) * t / 2
print("After", t, "seconds, the car will have traveled:", int(dx), "m.")

print("______________________________________")
# Exercise 3: What is the displacement on Mars?
print("Exercise 3: How far did Ryan fall on Mars?")
print("______________________________________")
t = float(input("How many seconds has Ryan been falling on Mars? "))
a = 3.71
v0 = 0
dx = v0 * t + .5 * a * t ** 2
print("After", t, "seconds of falling on Mars, Ryan's displacement will be:",
      int(dx), "m.")

print("______________________________________")
# Exercise 4: What is the final velocity on the Sun?
print("Exercise 4: What is Alex's final velocity on the Sun?")
print("______________________________________")
v0 = 0
a = 274
t = float(input("How long is Alex falling? "))
v = v0 + a * t
print("After", t, "seconds of falling on Mars, Alex's velocity will be:",
      int(v), "m/s!  That's really fast!")
