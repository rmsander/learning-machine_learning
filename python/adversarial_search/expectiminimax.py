import math

"""
Sources: 
    1. Geeks for Geeks's minimax article: https://www.geeksforgeeks.org/minimax-algorithm-in-game-theory-set-1-introduction/
    2. Wikipedia expectiminimax article: https://en.wikipedia.org/wiki/Expectiminimax
"""


def expectiminimax(node, depth, target_depth, scores, probabilities, turn="MAX"):
    """Expectiminimax implementation in Python."""

    # Base case
    if (depth == target_depth):
        return scores[node]

    # MAX (player turn)
    if turn == "MAX":
        alpha = -math.inf
        for child in children[node]:
            alpha = max(alpha, expectiminimax(child, depth + 1, target_depth, scores, probabilities, turn="MIN"))

    # MIN (adversarial AI turn)
    elif turn == "MIN":
        alpha = math.inf
        for child in children[node]:
            alpha = min(alpha, expectiminimax(child, depth + 1, target_depth, scores, probabilities, turn="CHANCE"))

    # Chance (expectation turn)
    elif turn == "CHANCE":
        alpha = 0
        for child in children[node]:
            alpha += probs[child] * expectiminimax(child, depth + 1, target_depth, scores, probabilities, turn="MAX")

    return alpha
