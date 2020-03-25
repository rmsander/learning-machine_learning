"""An interactive exercise for students to gain a stronger intuition with
conditional statements in python."""

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('Agg')

# Welcome to the Robot Guidance Program
print("Welcome to the Robot Guidance Program!")
print("Please use any of these directions: U,D,L,R")

initial_position = [0, 0]
positions = [[0, 0]]
num_moves = int(input("How many moves do you want the robot to make? "))
new_position = initial_position
for i in range(num_moves):
    print("This is move", i)
    print("Your current position is:", new_position)
    current_move = input(
        "For this move, which direction would you like the robot to move in? "
        "Please use one of U,D,L,R. ")
    if current_move == "U":
        # Only move up!
        positions.append([new_position[0], new_position[1] + 1])
        new_position[1] += 1
    elif current_move == "D":
        # Only move down!
        positions.append([new_position[0], new_position[1] - 1])
        new_position[1] -= 1
    elif current_move == "L":
        # Only move left!
        positions.append([new_position[0] - 1, new_position[1]])
        new_position[0] -= 1
    elif current_move == "R":
        # Only move right!
        positions.append([new_position[0] + 1, new_position[1]])
        new_position[0] += 1
    else:
        print("I'm sorry, please try entering that again.")
        num_moves += 1

print("All of your positions:", positions)
print("Your starting position was:", [0, 0])
print("Your final position was:", positions[-1])

plt.plot([positions[i][0] for i in range(len(positions))],
         [positions[i][1] for i in range(len(positions))])
plt.xlim((-num_moves, num_moves))
plt.ylim((-num_moves, num_moves))
plt.savefig('graph.png')
# ax = plt.axes()
plt.show()
