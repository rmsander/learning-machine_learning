import random
import math
import time
import numpy as np


class ExpectiminimaxAgent:
    """
    An example object-oriented implementation of the Expectiminimax algorithm.

    Inputs:
        1. probabilities (hash table/arr): Data structure capturing the transition probabilities of being in a state
            and taking an action: T: S X A --> P
        2. target_depth (float): Integer capturing the game tree depth for which we want to use heuristics to estimate
            the expected utility of different states.
    """

    def __init__(self, probabilities, target_depth):
        """Constructor for class."""

        # Attributes
        self.heuristics = ['monotonic', 'smoothness', 'free_tiles']  # Evaluation functions
        self.probabilities = probabilities  # Data structure capturing transition probability of T: S X A --> S
        self.target_depth = target_depth  # How far we want to search down the tree
        self.heuristic_weights = [1, 1, 1]  # For 3 heuristics

    def expectiminimax(self, node, depth, turn="MAX", turn_prev="CHANCE"):
        """
        Expectiminimax implementation in Python.  The procedure recursively alternates as follows:

                MAX AGENT --> CHANCE AGENT --> MIN AGENT --> CHANCE AGENT --> MAX AGENT ...

        Arguments:

            1. node (object): This should be a representation of state of the game tree, and should be callable by
                              the heuristic functions below.

            2. depth (int): Current depth in the game tree being considered.

            3. turn (str): String indicating the turn of the player, according to:

                                'MAX' --> The maximizing agent
                                'MIN' --> The minimizing agent
                                'CHANCE' --> The expectation calculation agent

            4. turn_prev (str): String indicating the previous turn of the player (used in CHANCE agent to decide who
                                takes the next turn).  Defined according to:

                                'MAX' --> The maximizing agent
                                'MIN' --> The minimizing agent
                                'CHANCE' --> The expectation calculation agent

        Returns:

            1. u (float): The maximum utility attainable by the maximizing agent given the adversarial behavior of the
                          minimizing agent.
        """

        # Base case
        if (depth == self.target_depth):  # Assumes we have to use heuristics and evaluation functions
            return self.compute_heuristics(node)

        # MAX (player turn)
        if turn == "MAX":
            u = -math.inf  # Placeholder value for final utility
            for child in self.find_child_states(node):  # Assumes MIN and MAX agents have same action space
                u = max(u, self.expectiminimax(child, depth + 1, turn="CHANCE", turn_prev="MAX"))

        # MIN (adversarial player turn)
        elif turn == "MIN":
            u = math.inf  # Placeholder value for final utility
            for child in self.find_child_states(node):  # Assumes MIN and MAX agents have same action space
                u = min(u, self.expectiminimax(child, depth + 1, turn="CHANCE", turn_prev="MIN"))

        # Chance (expectation turn)
        elif turn == "CHANCE":
            u = 0  # Placeholder value for final utility
            # Decide whether to do min or max next
            if turn_prev == "MIN":
                next_turn = "MAX"
            elif turn_prev == "MAX":
                next_turn = "MIN"

            # Now compute expected utility
            for child in self.find_child_states(node):  # # Assumes MIN and MAX agents have same action space
                u += self.probabilities[child] * self.expectiminimax(child, depth + 1, turn=next_turn, turn_prev="CHANCE")

        # Return estimate of value at the end
        return u


    def find_child_states(self, node):
        """
        Computes child states for the MAX block using the grid object API.
        Find all child states of a node that are accessible through a transition.

        Arguments:

            1. node (object): This should be a representation of state of the game tree, and should be callable by
                              the heuristic functions below.

        Returns:

            1. children (float): A heuristic-based estimate of a given node's value.
        """
        children = []
        # <YOUR CODE HERE>
        return chilren


    def compute_heuristics(self, node):  # Can be used as a sub-routine in expectiminimax to estimate value of search nodes
        """
        Computes heuristic values for estimating the utility of non-terminal elements in the game tree. If the state
        tree depth and branching factor are small enough, exact values of the terminal elements can instead be returned.

        Arguments:

            1. node (object): This should be a representation of state of the game tree, and should be callable by
                              the heuristic functions below.

        Returns:

            1. total (float): A heuristic-based estimate of a given node's value.

        Important note here: Only the relative ordering between the heuristics matter, not the absolute magnitudes.
        Can balance the heuristics using a set of coefficients.
        """
        total = 0
        if 'monotonic' in self.heuristics:
            total += self.heuristic_weights[0] * 1  # <YOUR CODE HERE> Compute monotonic heuristic
        if 'smoothness' in self.heuristics:
            total += self.heuristic_weights[1] * 1  # <YOUR CODE HERE> Compute smoothness heuristic
        if 'free_tiles' in self.heuristics:
            total += self.heuristic_weights[2] * 1  # <YOUR CODE HERE> Compute free tiles heuristic

        # Return the heuristic value at the end
        return total
