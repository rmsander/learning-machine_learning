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
        self.depth = 4
        self.heuristic_weights = [0.01, 0.01, 1, 0.1]  # For 4 heuristics
        self.heuristics = ['monotonic', 'smoothness', 'free_tiles', 'merges']
        self.time_to_move = 0.2  # seconds
        self.use_alpha_beta = True  # Whether to use alpha-beta
        self.probabilities = [0.9, 0.2]  # Probability table for 2 (0.9) and 4 (0.1)

        # Timing expectiminimax in getMove
        self.start_time = 0
        self.end_time = 0

        # Alpha-Beta pruning parameters
        self.ALPHA = -math.inf
        self.BETA = math.inf

    def getMove(self, grid):  # Chance (expectation turn)
        """
        Function for selecting a player's move using the Expectiminimax function
        and the grid object API.
        """
        self.start_time = time.time()  # Get start time for timing action selection
        # Call expectiminimax and return the best move
        u, best_move = self.expectiminimax(grid=grid, depth=0, turn="MAX",
                                           alpha=self.ALPHA, beta=self.BETA)
        self.end_time = time.time()  # Get end time for timing action selection
        print("Optimal expected utility for move {} is: {}".format(best_move, u))
        print("Time to compute: {} seconds".format(
            self.end_time - self.start_time))

        # Now return the best move
        return best_move

    def expectiminimax(self, grid, depth, turn="MAX", alpha=-math.inf, beta=math.inf):
        """
        Expectiminimax implementation in Python.  The procedure recursively
        alternates as follows:

            MAX AGENT --> MIN AGENT --> CHANCE AGENT --> MAX AGENT --> ...

        Arguments:

            1. grid (object): This should be a representation of state of the
               game tree, and should be callable by the heuristic functions
               below.

            2. depth (int): Current depth in the game tree being considered.

            3. turn (str): String indicating the turn of the player;

            Defined according to:

                                'MAX' --> The maximizing agent
                                'MIN' --> The minimizing agent
                                'CHANCE' --> The expectation calculation agent

            4. alpha (float): Maximum pruning value for alpha-beta pruning.

            5. beta (float): Minimum pruning value for alpha-beta pruning.

        Returns:

            1. u (float): The maximum utility attainable by the maximizing
            agent given the adversarial behavior of the
                          minimizing agent.

            2. best_move (int): (Returns on original recursive call) An integer
                                corresponding to the optimal move for the
                                MAX agent.
        """
        # Base case
        if (depth == self.depth):
            if turn == "CHANCE":
                return self.compute_heuristics(grid[1])  # Get heuristic approximation
            else:
                return self.compute_heuristics(grid)     # Get heuristic approximation

        # MAX (player turn)
        if turn == "MAX":
            u = -math.inf  # Placeholder value for final utility
            for move, child in enumerate(self.find_child_states_max(grid)):
                val = self.expectiminimax(child, depth + 1, turn="MIN",
                                          alpha=alpha, beta=beta)
                if val > u:  # We've found a new higher utility, so let's
                    # update it
                    u = val
                    if depth == 0:  # Deciding on the first move only
                        best_move = move  # Sets this move to be the optimal
                        # move -
                alpha = max(alpha, u)  # Update pruning parameter alpha
                if alpha >= beta:  # Prune by not considering other children
                    break

        # MIN (adversarial agent turn)
        elif turn == "MIN":
            u = math.inf  # Placeholder value for final utility
            for cell_index in self.find_child_states_min(grid):  # Children
                u = min(u, self.expectiminimax((cell_index, grid), depth + 1,
                                               turn="CHANCE", alpha=alpha, beta=beta))
                beta = min(beta, u)  # Update pruning parameter beta
                if alpha >= beta:  # Prune by not considering other children
                    break

        # CHANCE (expectation agent turn)
        elif "CHANCE" == turn:
            u = 0  # Placeholder value for final utility
            
            # Create probabilistic child nodes
            cell_index, grid = grid
            cell_2, cell_4 = copy.deepcopy(grid), copy.deepcopy(grid)
            cell_2.setCellValue(cell_index, 2)
            cell_4.setCellValue(cell_index, 4)
            children = [cell_2, cell_4]
            
            # Iterate through child nodes to compute expected utility value
            for i, child in enumerate(children):
                u += self.probabilities[i] * self.expectiminimax(child, depth + 1,
                                                turn="MAX", alpha=alpha, beta=beta)

        # Return estimate of value after recursive calls have finished
        if depth == 0:  # We're at the root grid
            if best_move is None:  # If for some reason we don't have one
                best_move = np.random.randint(low=0, high=3)  # Random action
            return u, best_move

        else:  # We're not at the root grid (return from a recursive call)
            return u

    def find_child_states_max(self, grid):
        """
        Computes child states for the MAX block using the grid object API.
        Find all child states of a grid that are accessible through a
        transition.

        Arguments:

            1. grid (object): This should be a representation of state of the
                              game tree, and should be callable by
                              the heuristic functions below.

        Returns:

            1. children (list): A list of the child Grid objects corresponding
                                to child grids of the game tree.
        """
        children = []  # Initialize output list of children
        for i in range(0, 4):  # Find children using up, right, left, and down
            child = copy.deepcopy(grid)
            can_move = child.move(i)  # Create a move, if possible
            if can_move:
                children.append(child)  # Add it to list of moves
        return children

    def find_child_states_min(self, grid):
        """
        Computes (probabilistic) child states for the MIN block using the grid
        object API.  Find all child states of a grid that are accessible
        through placing a new tile of value 2 or 4 on the 4 x 4 grid.

        Arguments:

            1. grid (object): This should be a representation of state of the
                              game tree, and should be callable by the 
                              heuristic functions below.

        Returns:

            1. children (list): A list of the child Grid objects corresponding
                                to child grids of the game tree.
            2. probabilities (list): A list of probabilities corresponding to
                                     the child states.  The probability of the
                                     child state children[i] is given by
                                     probabilities[i].
        """
        # Create output objects for child states and probabilities
        children = []

        # Find the available cells in the grid
        available_cells = grid.getAvailableCells()  # Returns a list of tuples

        # Iterate through all available cell indices and append to children
        for cell in available_cells:
            children.append(cell)
            
        return children

    def compute_heuristics(self, grid):
        """
        Computes heuristic values for estimating the utility of non-terminal
        elements in the game tree. If the state
        tree depth and branching factor are small enough, exact values of the
        terminal elements can instead be returned.

        Arguments:

            1. grid (object): This should be a representation of state of the
                              game tree, and should be callable by the 
                              heuristic functions below.

        Returns:

            1. total (float): A heuristic-based estimate of a given grid's
                              value.

        Important note here: Only the relative ordering between the
        heuristics matter, not the absolute magnitudes.
        Can balance the heuristics using a set of coefficients.
        """
        total = 0  # Set placeholder total value
        
        if 'monotonic' in self.heuristics:
            # Compute monotonic heuristic
            total += self.heuristic_weights[0] * self.monotone_heuristic(grid, use_rows=True, use_cols=True)
            
        if 'smoothness' in self.heuristics:
            # Compute smoothness heuristic
            total += self.heuristic_weights[1] * self.smoothness_heuristic(grid)

        if 'merges' in self.heuristics:
            # Compute merge heuristic
            total += self.heuristic_weights[2] * self.merges_heuristic(grid)

        if 'free_tiles' in self.heuristics:
            # Compute free tiles heuristic
            total += self.heuristic_weights[3] * self.free_tiles_heuristic(grid)

        return total

    def monotone_heuristic(self, grid, use_rows=True, use_cols=False):
        """
        Heuristic function that computes a heuristic based off of whether
        monotonicity is preserved between the rows and columns.

        Arguments:
            1. grid (object): This should be a representation of state of the
                              game tree, and should be callable by the 
                              heuristic functions below.

            2. use_rows (bool): Whether or not to look for monotone rows.

            3. use_cols (bool): Whether or not to look for monotone columns.

        Returns:
            1. total (float): The total reward (in this case penalty).
        """
        total = 0  # Set placeholder heuristic value

        # Look at monotonicity over rows
        if use_rows:
            for i in range(4):  # Iterate through rows of board
                row_tiles = [grid.getCellValue((i, j)) for j in range(4)]  # Values along a row
                sorted_rows = copy.deepcopy(row_tiles)
                sorted_rows.sort()  # Sort in increasing order
                if not (row_tiles == sorted_rows or row_tiles == sorted_rows[::-1]):
                    max_tile_index = np.argmax(row_tiles)
                    if max_tile_index != 0 and max_tile_index != 3:  # Not corner
                        total -= abs(row_tiles[max_tile_index] - row_tiles[max_tile_index - 1]) + \
                                 abs(row_tiles[max_tile_index] - row_tiles[max_tile_index + 1])

        if use_cols:
            for j in range(4):  # Iterate through rows of board
                col_tiles = [grid.getCellValue((i, j)) for i in range(4)]  # Values along a column
                sorted_cols = copy.deepcopy(col_tiles)
                sorted_cols.sort()  # Sort in increasing order
                if not (col_tiles == sorted_cols or col_tiles == sorted_rows[::-1]):
                    max_tile_index = np.argmax(col_tiles)
                    if max_tile_index != 0 and max_tile_index != 3:  # Not corner
                        total -= abs(col_tiles[max_tile_index] - col_tiles[max_tile_index - 1]) + \
                                 abs(col_tiles[max_tile_index] - col_tiles[max_tile_index + 1])

        return total

    # Reference: https://home.cse.ust.hk/~yqsong/teaching/comp3211/projects
    # /2017Fall/G11.pdf
    def smoothness_heuristic(self, grid):
        """
        Heuristic function that computes a heuristic based off of the
        differences between adjacent tiles.

        Arguments:
            1. grid (object): This should be a representation of
                              state of the game tree, and should be callable 
                              by the heuristic functions below.

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
                    if (neighbor[0] < 0 or neighbor[0] > 3 or neighbor[1] < 0 or neighbor[1] > 3):
                        continue
                    if (neighbor, (i, j)) in existing_pairs.keys():  # Already checked
                        continue
                    else:  # If we haven't seen it, add it
                        existing_pairs[((i, j), neighbor)] = 1
                        if value == grid.getCellValue(neighbor):
                            total -= abs(grid.getCellValue(neighbor) - value)
        return total

    # Reference: https://www.robertxiao.ca/hacking/2048-ai/
    def merges_heuristic(self, grid):
        """
        Heuristic function that computes a heuristic based off of how many
        pairwise merges can be performed.

        Arguments:
            1. grid (object): This should be a representation of
                              state of the game tree, and should be callable 
                              by the heuristic functions below.

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
                    if (neighbor[0] < 0 or neighbor[0] > 3 or neighbor[1] < 0 or neighbor[1] > 3):
                        continue
                    if (neighbor, (i, j)) in existing_pairs.keys():  # Already checked
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
                              state of the game tree, and should be callable 
                              by the heuristic functions below.

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

