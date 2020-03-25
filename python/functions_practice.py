"""Quick fill-in-the-blank exercises for students to practice with functions
in Python.  Main emphasis is on revisiting pizza function from before."""

# Let's redo our pizza party using functions!

# Step 1: Define function + inputs
def pizza_party():
    toppings = [str(x) for x in input(
        "What ingredients do you want on your pizza? Please enter as words "
        "separated by commas. ").split(',')]
    # Step 2: Write function content!
    total = 5  # Base cost of 5
    for topping in toppings:
        if topping == "cheese":
            total += 1  # JD
        elif topping == "bacon":
            total += 1  # JD
        elif topping == "mushroom":
            total += 1  # JD
        elif topping == "pineapple":
            total += 1  # JD
        elif topping == "chicken":
            total += 1  # JD
        elif topping == "fish":
            total += 1  # JD
        elif topping == "banana":
            total += 1  # JD
        elif topping == "chocolate":
            total += 1  # JD
        elif topping == "birds":
            total += 1  # JD
        elif topping == "eggs":
            total += 1  # JD
        elif topping == "pancakes":
            total += 1  # JD
        elif topping == "bullets":  # FOR TOUGH GUYS
            total += 1  # JD
        elif topping == "hommos":
            total += 1  # JD
        elif topping == "falafel":
            total += 1  # JD
        elif topping == "mansaf":
            total += 1  # JD
        else:
            print("Ok, we'll add", topping, "to your pizza :)")
            total += 1  # JD
    # Step 3: Return our total cost of the pizza!
    return total


# cost = pizza_party()
# print("The total cost of your pizza is:",cost)


# Let's figure out if a number is prime! (Basis of RSA Encryption)

# Step 1: Define function and parameters
def prime(n):
    # Step 2: Write body of code
    for i in range(2, n):
        if n % i == 0:
            # Step 3: Return y-value
            # Not prime
            return False
    # Step 3: Return y-value
    # Number is prime
    return True


primes = []
for i in range(2, 10000):
    if prime(i) == True:
        primes.append(i)
# print("Here is a list of prime numbers:",primes)


# Now let's break RSA Encryption!
# Step 1: Define function and parameters
def prime_factor(n):
    # Step 2: Body of the code n -> T/F (boolean)
    for i in range(2, n):  # Outer for loop
        for j in range(2, n):  # Inner for loop
            if prime(i) == True and prime(j) == True and i * j == n:
                print("i = ", i, "and j =", j)
                return True  # Yes, possible
    return False  # Not possible

# Output list
possible = []
for k in range(2, 250):
    if prime_factor(k) == True:
        possible.append(k)
print(possible)
