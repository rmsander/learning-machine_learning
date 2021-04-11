"""This class is a basic implementation of a Dutch Auction.  This specific
implementation looks at having a Uniform bidder and a Quasi-Hyperbolic discounting
bidder compete against one another."""

import numpy as np
import copy

class Uniform:
    """Class for a player with uniform discounting."""
    def __init__(self, starting_value=0):
        self.value = starting_value

    def reset_value(self, starting_price):
        """Resets the value of the Uniform bidder to be uniformly over the
        domain [0, 1.1 x starting_price]."""
        upper_bound = int(starting_price * 1.1)
        self.value = np.random.randint(low=0, high=upper_bound)

class QuasiHyperbolic:
    """Class for a player with quasi-hyperbolic discounting."""
    def __init__(self, beta, delta, value):
        self.beta = beta  # Long-term discount factor
        self.delta = delta  # Short-term discount factor
        self.value = value  # The value the player holds for an item

    def compute_utility(self, t, price_delta):
        """Computes the utility for a given time."""
        # Compute "periods to bid"
        gamma = int(t - (self.value / price_delta))  # Number of periods until price = value

        # Compute the "positive" utility
        first_term = (self.beta * self.delta ** (gamma)) * self.value

        # Compute the expected cost ("negative" utility) of losing the item, which has value -self.value
        second_term = sum([1 / (t - tau) * self.beta * self.delta ** (t-tau) for tau in range(gamma)]) * self.value

        # Take final utility as first_term - second_term
        return first_term - second_term


    def reset_value(self, starting_price):
        """Resets the value of the Uniform bidder to be uniformly over the
        domain [0, 1.1 x starting_price]."""
        upper_bound = int(starting_price * 1.1)
        self.value = np.random.randint(low=0, high=upper_bound)


class DutchAuction:
    """Class implementing a two-player Dutch Auction system."""
    def __init__(self, starting_price, p1, p2, price_increment=1):
        self.starting_price = starting_price  # Starting price
        self.price_increment = price_increment  # How much price drops by
        self.T = self.starting_price // self.price_increment  # How many periods until bid price hits zero
        self.p1 = p1  # For now, this can be a QuasiHyperbolic() class
        self.p2 = p2  # For now, this can be a Constant() class

        # Keep track of results from bidding
        self.player_bids = []
        self.bid_prices = []

    def run_action(self):
        """Method to run a single action simulation."""
        # Reset the value of the random player
        self.p2.reset_value(self.starting_price)

        # Set a variable for price using the starting price
        current_price = copy.deepcopy(self.starting_price)
        t = copy.deepcopy(self.T)
        while current_price > 0:  # Bid does not go below zero

            # If True, p2 places bid
            if self.p2.value >= current_price:
                self.player_bids.append(2)
                self.bid_prices.append(current_price)
                return

            # If True, p1 places bid
            if self.p1.compute_utility(t, self.price_increment) >= current_price:
                self.player_bids.append(1)
                self.bid_prices.append(current_price)
                return

            # No one has placed bid, decrement price and number of periods remaining
            current_price -= self.price_increment
            t -= 1

        # If here, no one has placed bid
        self.player_bids.append(-1)
        self.bid_prices.append(-1)


    def run_auction_simulations(self, n=1000):
        """Makes repeated calls to run_action to run Monte Carlo simulations."""
        for i in range(n):  # Number of simulations
            self.run_action()


def main():
    # Set parameters
    BETA = 0.5
    DELTA = 0.9
    P1_VALUE = 7
    P2_VALUE = 5
    STARTING_PRICE = 10
    PRICE_INCREMENT = 1

    # Make players
    p1 = QuasiHyperbolic(BETA, DELTA, P1_VALUE)
    p2 = Uniform(starting_value=P2_VALUE)

    # Make auction
    auction = DutchAuction(STARTING_PRICE, p1, p2, price_increment=PRICE_INCREMENT)

    # Run 10 auctions
    auction.run_auction_simulations(n=1000)

    print("Player bids: \n{}".format(auction.player_bids))
    print("Bid prices: \n{}".format(auction.bid_prices))


if __name__ == '__main__':
    main()