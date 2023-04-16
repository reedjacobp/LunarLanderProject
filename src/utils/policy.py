import numpy as np

class EpsilonGreedyPolicy:
    # def __init__(self):
    #     # Use this as needed

    # Select action index based on epsilon-greedy policy
    def select_action(self, s, actions, Q, epsilon):
        if np.random.rand() < epsilon:
            # Be random with a probability of epsilon
            return np.random.randint(0, len(actions) - 1)
        else:
            # Be greedy with a probability of 1-epsilon
            return np.argmax([Q[(s, a)] for a in actions])
