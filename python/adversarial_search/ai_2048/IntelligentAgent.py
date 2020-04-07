import random
import math
import time
import numpy as np
import copy
from BaseAI import BaseAI


class IntelligentAgent(BaseAI):
    """
    Inputs:
      1. probabilities (hash table/arr): Data structure capturing the
      transition probabilities of being in a state
          and taking an action: T: S X A --> P
      2. target_depth (float): Integer capturing the game tree depth for
      which we want to use heuristics to estimate
          the expected utility of different states.
      3. use_alpha_beta (bool): Boolean for whether or not we use alpha-beta
      pruning.
    """
    def __init__(self):

        # Inherit from superclass
        super(IntelligentAgent, self).__init__()

        # Put attributes here - Some ideas are below in block comment
        self.target_depth = 4
        self.heuristic_weights = [1, 1, 1, 1]  # For 4 heuristics
        self.heuristics = ['merges'] #['monotonic', 'smoothness', 'free_tiles', 'merges']
        self.time_to_move = 0.2  # seconds
        self.use_alpha_beta = True  # Whether to use alpha-beta

        # pruning in expectiminimax
        self.start_time = 0
        self.end_time = 0

        # Alpha-Beta pruning parameters
        self.ALPHA = -math.inf
        self.BETA = math.inf

    def getMove(self, grid):  # Chance (expectation turn)
        """
        Recommendation: This is likely where you'll want to implement the 90%
        probability of 2 and 10% probability of 4.
        Maybe the computer chooses the tile randomly?  Or maybe this is
        designed to make the player lose.
        """
        self.start_time = time.time()  # Get start time for timing action selection
        # Call expectiminimax and return the best move
        u, best_move = self.expectiminimax(node=grid, depth=0, turn="MAX",
                                            alpha=self.ALPHA, beta=self.BETA)
        self.end_time = time.time()  # Get end time for timing action selection
        print("Optimal expected utility for move {} is: {}".format(best_move, u))
        print("Time to compute: {} seconds".format(self.end_time-self.start_time))

        # Now return the best move
        return best_move

    def expectiminimax(self, node, depth, turn="MAX", turn_prev="CHANCE",
                       alpha=1, beta=1):
        """
        Expectiminimax implementation in Python.  The procedure recursively
        alternates as follows:

                MAX AGENT --> CHANCE AGENT --> MIN AGENT --> CHANCE AGENT -->
                MAX AGENT ...

        Arguments:

            1. node (object): This should be a representation of state of the
            game tree, and should be callable by the heuristic functions below.

            2. depth (int): Current depth in the game tree being considered.

            3. turn (str): String indicating the turn of the player;

            Defined according to:

                                'MAX' --> The maximizing agent
                                'MIN' --> The minimizing agent
                                'CHANCE' --> The expectation calculation agent

            4. turn_prev (str): String indicating the previous turn of the
            player (used in CHANCE agent to decide who takes the next turn).

            Defined according to:

                                'MAX' --> The maximizing agent
                                'MIN' --> The minimizing agent
                                'CHANCE' --> The expectation calculation agent

            5. alpha (float): Maximum pruning value for alpha-beta pruning.

            6. beta (float): Minimum pruning value for alpha-beta pruning.

        Returns:

            1. u (float): The maximum utility attainable by the maximizing
            agent given the adversarial behavior of the
                          minimizing agent.
        """
        # Base case
        if (depth == self.target_depth):
            return self.compute_heuristics(node)  # Get heuristic approximation

        # MAX (player turn)
        if turn == "MAX":
            u = -math.inf  # Placeholder value for final utility
            for move, child in enumerate(self.find_child_states_max(node)):
                val = self.expectiminimax(child, depth + 1, turn="MIN",
                                          turn_prev="MAX", alpha=alpha,
                                          beta=beta)
                if val > u:  # We've found a new higher utility, so let's update it
                    u = val
                    if depth == 0:  # Deciding on the first move only
                        best_move = move  # Sets this move to be the optimal move -
                alpha = max(alpha, u)  # Update pruning parameter alpha
                if alpha >= beta:  # Prune by not considering other children
                    break

        # MIN (adversarial agent turn)
        elif turn == "MIN":
            u = math.inf  # Placeholder value for final utility
            for child in self.find_child_states_min(node)[0]:  # Children
                u = min(u, self.expectiminimax(child, depth + 1, turn="CHANCE",
                                               turn_prev="MIN", alpha=alpha,
                                               beta=beta))
                beta = min(beta, u)  # Update pruning parameter beta
                if alpha >= beta:  # Prune by not considering other children
                    break

        # CHANCE (expectation agent turn)
        elif "CHANCE" == turn:
            u = 0  # Placeholder value for final utility

            """
            NOTE: I don't think we actually need this block, since there's no
                  stochasticity of taking an action for the MAX agent (i.e. 
                  whatever action you select yields the new state).
            """
            """
            # If previous turn was max, the next turn will be min
            if turn_prev == "MAX":
                next_turn = "MIN"
                children = self.find_child_states_max(
                    node)  # Find child states for agent
                for child in children:  # For if previous turn was max
                    u += (1 / len(children)) * self.expectiminimax(child,
                                                                   depth + 1,
                                                                   turn=next_turn,
                                                                   turn_prev="CHANCE",
                                                                   alpha=alpha,
                                                                   beta=beta)
            """
            # If previous turn was min, the next turn will be max
            if turn_prev == "MIN":
                next_turn = "MAX"
                children, probabilities = self.find_child_states_min(node)
                for child, prob in zip(children, probabilities):
                    u += prob * min(u, self.expectiminimax(child, depth + 1,
                                                           turn=next_turn,
                                                           turn_prev="MIN",
                                                           alpha=alpha,
                                                           beta=beta))

        # Return estimate of value at the end
        if depth == 0:  # We're at the root node
            return u, best_move

        else:  # We're not at the root node (return from a recursive call)
            return u

    def insertRandomTiles(self, numTiles: int):
        """ Insert numTiles number of random tiles. For initialization. """
        for i in range(numTiles):
            tileValue = self.getNewTileValue()
            cells = self.grid.getAvailableCells()
            cell = random.choice(cells) if cells else None
            self.grid.setCellValue(cell, tileValue)

    def find_child_states_max(self, grid):
        """
        Computes child states for the MAX block using the grid object API.
        Find all child states of a node that are accessible through a
        transition.

        Arguments:

            1. grid (object): This should be a representation of state of the
            game tree, and should be callable by
                              the heuristic functions below.

        Returns:

            1. children (list): A list of the child Grid objects corresponding
                                to child nodes of the game tree.
        """
        children = []  # Initalize output list of children
        for i in range(0, 4):  # Find children using up, right, left, and down
            child = copy.copy(grid)
            can_move = child.move(i)  # Create a move, if possible
            if can_move:
                children.append(child)  # Add it to list of moves
        return children

    def find_child_states_min(self, grid):
        """
        Computes (probabilistic) child states for the MIN block using the grid
        object API.  Find all child states of a node that are accessible
        through placing a new tile of value 2 or 4 on the 4 x 4 grid.

        Arguments:

            1. grid (object): This should be a representation of state of the
            game tree, and should be callable by the heuristic functions below.

        Returns:

            1. children (list): A list of the child Grid objects corresponding
                                to child nodes of the game tree.
            2. probabilities (list): A list of probabilities corresponding to
                                     the child states.  The probability of the
                                     child state children[i] is given by
                                     probabilities[i].
        """
        # Create output objects for child states and probabilities
        children = []
        probabilities = []

        # Find the available cells in the grid
        available_cells = grid.getAvailableCells()  # Returns a list of tuples
        n_avail = len(available_cells)

        # Iterate through all available cells
        for cell in available_cells:
            for tile, t_prob in [(2, .9), (4, .1)]:  # Choices for tile

                # Probability for a child state
                p = (1 / n_avail) * t_prob

                # Make a child node
                child = copy.deepcopy(grid)     # Create copy of node
                child.setCellValue(cell, tile)  # Add new element
                children.append(child)          # Add to list of children
                probabilities.append(p)         # Add to list of probabilities
        return children, probabilities

    def compute_heuristics(self, node):  # Used as expectiminimax sub-routine
        """
        Computes heuristic values for estimating the utility of non-terminal
        elements in the game tree. If the state
        tree depth and branching factor are small enough, exact values of the
        terminal elements can instead be returned.

        Arguments:

            1. node (object): This should be a representation of state of the
            game tree, and should be callable by the heuristic functions below.

        Returns:

            1. total (float): A heuristic-based estimate of a given node's
            value.

        Important note here: Only the relative ordering between the
        heuristics matter, not the absolute magnitudes.
        Can balance the heuristics using a set of coefficients.
        """
        total = 0  # Set placeholder total value
        if 'monotonic' in self.heuristics:
            # Compute monotonic heuristic
            total += self.heuristic_weights[0] * self.monotone_heuristic(node, use_rows=True, use_cols=False)

        if 'smoothness' in self.heuristics:
            # Compute smoothness heuristic
            total += self.heuristic_weights[1] * self.smoothness_heuristic(node)

        if 'merges' in self.heuristics:
            # Compute merge heuristic
            total += self.heuristic_weights[2] * self.merges_heuristic(node)

        if 'free_tiles' in self.heuristics:
            # Compute free tiles heuristic
            total += self.heuristic_weights[3] * self.free_tiles_heuristic(node)

        # Return the heuristic value at the end
        return total

    def monotone_heuristic(self, grid, use_rows=True, use_cols=False):
        """
        Heuristic function that computes a heuristic based off of whether
        monotonicity is preserved between the rows and columns.

        Arguments:
            1. grid (object): This should be a representation of state of the
            game tree, and should be callable by the heuristic functions below.

            2. use_rows (bool): Whether or not to look for monotone rows.

            3. use_cols (bool): Whether or not to look for monotone columns.

        Returns:
            1. total (float): The total reward (in this case penalty).
        """
        total = 0  # Set placeholder heuristic value

        # Look at monotonicity over rows
        if use_rows:
            for i in range(4):  # Iterate through rows of board
                row_tiles = [grid.getCellValue((i, j)) for j in
                             range(4)]  # Values along a row
                sorted_rows = copy.deepcopy(row_tiles)
                sorted_rows.sort()  # Sort in increasing order
                if not (row_tiles == sorted_rows or row_tiles == sorted_rows[
                                                                 ::-1]):  #
                    # Check if row monotone increasing or decreasing
                    max_tile_index = np.argmax(row_tiles)
                    try:  # Case if max not on corners
                        total -= abs(row_tiles[max_tile_index] - row_tiles[
                            max_tile_index - 1]) + abs(row_tiles[max_tile_index] - row_tiles[
                                max_tile_index + 1])
                    except:  # If max is on the corners, it should be ok
                        total -= 0

        # Look at monotonicity over columns
        if use_cols:
            for i in range(4):  # Iterate through rows of board
                col_tiles = [grid.getCellValue((j, i)) for j in
                             range(4)]  # Values along a row
                sorted_cols = copy.deepcopy(col_tiles)
                sorted_cols.sort()  # Sort in increasing order
                if not (col_tiles == sorted_cols or col_tiles == sorted_cols[
                                                                 ::-1]):  #
                    # Check if row monotone increasing or decreasing
                    max_tile_index = np.argmax(col_tiles)
                    try:  # Case if max not on corners
                        total -= abs(col_tiles[max_tile_index] - col_tiles[
                            max_tile_index - 1]) + abs(
                            col_tiles[max_tile_index] - col_tiles[
                                max_tile_index + 1])
                    except:  # If max is on the corners, it should be ok
                        total -= 0

        return total

    # Reference: https://home.cse.ust.hk/~yqsong/teaching/comp3211/projects/2017Fall/G11.pdf
    def smoothness_heuristic(self, grid):
        """
        Heuristic function that computes a heuristic based off of the
        differences between adjacent tiles.

        Arguments:
            1. grid (object): This should be a representation of
            state of the game tree, and should be callable by the heuristic
            functions below.

        Returns:
            1. total (float): The total reward (in this case penalty).
        """
        existing_pairs = {}  # Keep track of pairs to avoid computing again
        total = 0  # Set placeholder heuristic value
        for i in range(4):  # Iterate through rows of board
            for j in range(4):  # Iterate through columns of board
                value = grid.getCellValue((i, j))
                neighbor_indices = [(i - 1, j), (i + 1, j),
                                    (i, j - 1), (i, j + 1)]
                for neighbor in neighbor_indices:
                    if ((i, j), neighbor) in existing_pairs.keys(): # Already checked
                        continue
                    else:  # If we haven't seen it, add it
                        existing_pairs[((i, j), neighbor)] = 1
                        try:  # Neighbor does exist
                            total -= abs(grid.getCellValue(neighbor) - value)
                        except:
                            total -= 0  # Neighbor does not exist
        return total

    # Reference: https://www.robertxiao.ca/hacking/2048-ai/
    def merges_heuristic(self, grid):
        """
        Heuristic function that computes a heuristic based off of how many
        pairwise merges can be performed.

        Arguments:
            1. grid (object): This should be a representation of
            state of the game tree, and should be callable by the
            heuristic functions below.

        Returns:
            1. total (float): The total reward, in this case the total number
                              of merges that can be performed.
        """
        existing_pairs = {}
        total = 0  # Set placeholder heuristic value
        for i in range(4):  # Iterate through rows of board
            for j in range(4):  # Iterate through columns of board
                value = grid.getCellValue((i, j))  # Get value
                neighbor_indices = [(i - 1, j), (i + 1, j),
                                    (i, j - 1), (i, j + 1)]  # Get neighbors
                for neighbor in neighbor_indices:  # Iterate through neighbors
                    if ((i, j), neighbor) in existing_pairs.keys():  # Already checked
                        continue
                    else:  # If we haven't seen it, add it
                        existing_pairs[((i, j), neighbor)] = 1
                        if grid.getCellValue(neighbor) == value:
                            total += 1
        return total

    def free_tiles_heuristic(self, grid):
        """
        Heuristic function that computes a heuristic based off of how many
        the number of free tiles on the board.

        Arguments:
            1. grid (object): This should be a representation of
            state of the game tree, and should be callable by the
            heuristic functions below.

        Returns:
            1. total (float): The total reward, in this case the number of
                              free tiles on the board.
        """
        total = 0  # Set placeholder heuristic value
        for i in range(4):  # Iterate through rows of board
            for j in range(4):  # Iterate through columns of board
                if grid.canInsert((i, j)):  # Checks if assigned value is 0
                    total += 1  # Adds a score of 1 for each free tile

        return total
