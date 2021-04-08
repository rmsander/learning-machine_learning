"""A game-style exercise where students can practice with conditional
statements in Python.  Students use input and if/else statements to compute
how much a pizza costs by selecting certain ingredients."""

#Pizza Maker
print("Welcome to Pizza Maker!")
print("The ingredients you can choose from: ")
print("__________________________________________________________")
print("cheese, peppers, sausage, pineapple, chicken, and broccoli")
print("__________________________________________________________")

#Now let's find the cost of the pizza!
print("Now let's find the cost of the pizza!")

total = 0
base = 5
total += base

cheese = input("Does your pizza have cheese? ")
if cheese == "yes":
    total += 1

peppers = input("Does your pizza have peppers? ")
if peppers == "yes":
    total += 1.75

sausage = input("Does your pizza have sausage? ")
if sausage == "yes":
    total += 2

pineapple = input("Does your pizza have pineapple? ")
if pineapple == "yes":
    total += 3

chicken = input("Does your pizza have chicken? ")
if chicken == "yes":
    total += 1.5

broccoli = input("Does your pizza have broccoli? ")
if broccoli == "yes":
    total += .25

print("The total cost of your pizza is:",total,"JDs.")


#What if we want more than one pizza?
number_of_pizzas = float(input("How many pizzas would you like to order? "))
total_cost = total * number_of_pizzas
print("The total cost of your pizzas is:",total_cost,"JDs.")

