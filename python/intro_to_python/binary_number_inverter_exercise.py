"""Fun exercise for teaching students about dictionaries and binary numbers."""

# Let's make a binary number inverter
print("Let's make a binary number inverter with dictionaries!")

# Step 1: Initialize our empty dictionary to convert numbers!
converter = ?

# Step 2: Write our key/value pairs for 1 and 0!
converter[?] = ?
converter[?] = ?

# Step 3: Test!
inverted_binary_number = []
user_input = input("What is your binary number? ")
binary_number = list(user_input)
for number in binary_number:
    inverted_binary_number.append(converter[int(number)])

# We could also do this this way!
# for i in range(len(binary_number)):
# inverted_binary_number.append(converter[int(binary_number[i])])

final_number = ""
for number in inverted_binary_number:
    final_number = final_number + str(number)
print("Your original number was:", user_input, "and your inverted number is:",
      final_number)
