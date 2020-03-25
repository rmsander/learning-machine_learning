"""An exercise for students to work with ciphers, which can help to
simultaneously teach the ideas of encoding/decoding with dictionaries."""

# Import this package
import string

# Welcome to the encoding and decoding cipher!
print("Welcome to the encoding and decoding cipher!")
alphabet = list(string.ascii_lowercase) + list(string.ascii_uppercase)
encode_cipher = {}
for i in range(len(alphabet)):
    if i != len(alphabet) - 1:
        encode_cipher[alphabet[i]] = alphabet[i + 1]
    else:
        encode_cipher[alphabet[i]] = alphabet[0]
encode_cipher[" "] = " "
encode_cipher["!"] = "?"
encode_cipher["?"] = "!"
encode_cipher[","] = "."
encode_cipher["."] = ","

message = list(input("Please type your secret message here: "))
message.reverse()
encoded_message = ""
for i in range(len(message)):
    encoded_message += encode_cipher[message[i]]

print("Your secret message was encoded as:", encoded_message)
# Now it's time to decode our message!
encoded_message = list(encoded_message)
encoded_message.reverse()
decode_cipher = {}
for i in range(len(alphabet)):
    if i != 0:
        decode_cipher[alphabet[i]] = alphabet[i - 1]
    else:
        decode_cipher[alphabet[i]] = alphabet[len(alphabet) - 1]
decode_cipher[" "] = " "
decode_cipher["!"] = "?"
decode_cipher["?"] = "!"
decode_cipher[","] = "."
decode_cipher["."] = ","
decoded_message = ""
for i in range(len(encoded_message)):
    decoded_message += decode_cipher[encoded_message[i]]

print("We decoded the secret message!")
print("Your decoded message is:", decoded_message)
