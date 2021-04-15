"""Classes for DutchAuctions, Uniform Bidders, and Quasi-Hyperbolic bidders.
Capable of running Monte Carlo simulations."""

import numpy as np
import copy
import pandas as pd

"""
Assumptions we make:
    
    1. f a player loses, they experience a utility of -v (the negative of the 
        value they hold for a product.
        
    2. We assume that Player 1 knows that Player 2 is a uniform bidder.  This 
        affects the expected utilities for the quasi-hyperbolic discounting bidder.
    
    3. Player 1 is a uniform bidder.
"""

class UniformBidder:
    """Class for a player with uniform discounting.

    Parameters:
        starting_value (float):  The starting value the player holds for the
            item they are bidding on.
    """
    def __init__(self, starting_value=0):
        self.value = starting_value

    def reset_value(self, starting_price):
        """Resets the value of the UniformBidder bidder to be uniformly over the
        domain [0, 1.1 x starting_price].

        Parameters:
            starting_price (float): The starting price of the item to be bid for.
                Used for scaling the chosen starting value for the player.
        """
        upper_bound = int(starting_price * 1.1)
        self.value = np.random.randint(low=0, high=upper_bound)

class QuasiHyperbolicBidder:
    """Class for a bidding player with quasi-hyperbolic discounting.

    Parameters:
        beta (float):  A float value (typically) between [0, 1] representing the
            long-term discount factor used for quasi-hyperbolic discounting.
        delta (float):  A float value (typically) between [0, 1] representing the
            short-term, exponential discount factor used for quasi-hyperbolic
            discounting.
        value (float): The undiscounted, deterministic value the player holds for
            the item being bid on.
    """
    def __init__(self, beta, delta, value):
        self.beta = beta  # Long-term discount factor
        self.delta = delta  # Short-term discount factor
        self.value = value  # The value the player holds for an item
        self.optimal_utilities = {}
        self.optimal_periods = {}

    def compute_utility(self, t, tau, price, price_delta):
        """Computes the utility for a given time.

        Parameters:
            t (int):  The current time step for bidding.
            tau (int): The number of timesteps into the future considered.
            price (float):  The current price of the item for which the player can bid.
            price_delta (float): How much the price is decremented during each period.

        Returns:
            u (float): The expected, discounted utility of placing a bid at timestep
                t + tau, with price given by price and the change in price given by
                price_delta.  Note that this is computed with respect to the current
                time step t.
        """
        # Probability of being able to bid tau steps into the future
        prob_bid = np.prod([1-(1/(1-price + price_delta*i)) for i in range (1,tau+1)])

        # Probability of not being able to bid tau steps into the future
        prob_otherbid = np.sum([1/(1-price+price_delta*j)*np.prod([(1/(1-price+price_delta*i)) for i in range(1, j+1)]) for j in range(1, tau+1)])

        # Quasi-hyperbolic discounting factor
        time_discount = self.beta*self.delta**tau

        # The undiscounted, deterministic instantaneous utility of bidding at time step t + tau
        bid_utility = self.value-(price-price_delta*(t+tau))

        # Take final utility as first_term - second_term
        return (prob_bid * time_discount * bid_utility) - (prob_otherbid * time_discount * self.value)

    def reset_value(self, starting_price):
        """Resets the value of the UniformBidder bidder to be uniformly over the
        domain [0, 1.1 x starting_price].

        Parameters:
            starting_price (float): The starting price of the item to be bid for.
                Used for scaling the chosen starting value for the player.
        """
        upper_bound = int(starting_price * 1.1)
        self.value = np.random.randint(low=0, high=upper_bound)


class DutchAuction:
    """Class implementing a two-player Dutch Auction system.

    Parameters:
        starting_price (float):  The starting price of the item to be bid on.
        p1 (QuasiHyperbolicBidder):  Player 1 for the Dutch Auction.
        p2 (ConstantBidder):  Player 2 for the Dutch Auction.
        price_increment (float): The amount by which the price changes after
            each time step.  Defaults to 1.0
    """
    def __init__(self, starting_price, p1, p2, price_increment=1.0):
        self.starting_price = starting_price  # Starting price
        self.price_increment = price_increment  # How much price drops by
        self.T = int(self.starting_price // self.price_increment)  # How many periods until bid price hits zero
        self.p1 = p1  # For now, this can be a QuasiHyperbolicBidder() class
        self.p2 = p2  # For now, this can be a Constant() class

        # Keep track of results from bidding
        self.player_bids = []
        self.bid_prices = []
        self.player_vals = []

    def run_auction(self):
        """Method to run a single action simulation."""
        # Reset the value of the random player
        self.p2.reset_value(self.starting_price)
        self.p1.reset_value(self.starting_price)
        self.player_vals.append(self.p1.value)

        # Set a variable for price using the starting price
        current_price = copy.deepcopy(self.starting_price)
        steps_left = copy.deepcopy(self.T)
        t = 0
        
        while current_price > 0:  # Bid does not go below zero

            # If True, p2 places bid
            if self.p2.value >= current_price:
                self.player_bids.append(2)
                self.bid_prices.append(current_price)
                return

            # If True, p1 places bid
            utilities = [self.p1.compute_utility(t, tau, current_price, self.price_increment) for tau in range(steps_left-1)]

            # Find optimal utility and store it
            self.p1.optimal_utilities[t] = np.max(utilities)
            self.p1.optimal_periods[t] = np.argmax(utilities) + t

            # Check if best time to bid is at current time step
            if np.argmax(utilities) == 0:  # Best time to bid is now
                df_dict = [self.p1.optimal_utilities, self.p1.optimal_periods]
                df = pd.DataFrame(df_dict)
                df.to_csv("best_utilities.csv")
                self.player_bids.append(1)
                self.bid_prices.append(current_price)
                return

            # No one has placed bid, decrement price and number of periods remaining
            current_price -= self.price_increment
            steps_left -= 1
            t += 1

        # If here, no one has placed bid
        self.player_bids.append(-1)
        self.bid_prices.append(-1)

    def run_auction_simulations(self, n=1000):
        """Makes repeated calls to run_action to run Monte Carlo simulations."""
        for i in range(n):  # Number of simulations
            self.run_auction()


def main():
    # Set parameters
    BETA = 1
    DELTA = .5
    P1_VALUE = 7
    P2_VALUE = 5
    STARTING_PRICE = 10
    PRICE_INCREMENT = 1

    # Make players
    p1 = QuasiHyperbolicBidder(BETA, DELTA, P1_VALUE)
    p2 = UniformBidder(starting_value=P2_VALUE)

    # Make auction
    auction = DutchAuction(STARTING_PRICE, p1, p2, price_increment=PRICE_INCREMENT)

    # Run 10 auctions
    auction.run_auction_simulations(n=10)

    # Extract player bids, player values, and bid prices
    auction.player_bids = np.array(auction.player_bids)
    auction.player_vals = np.array(auction.player_vals)
    auction.bid_prices = np.array(auction.bid_prices)

    # Display bids and prices
    print("Player bids: \n{}".format(auction.player_bids))
    print("Bid prices: \n{}".format(auction.bid_prices))

    # Count the number of times QuasiHyperbolicBidder bids early
    count_bid_early = sum(auction.bid_prices[auction.player_bids==1] < auction.player_vals[auction.player_bids==1])
    print("Number of times QuasiHyperbolicBidder bids early: {}".format(count_bid_early))


if __name__ == '__main__':
    main()
