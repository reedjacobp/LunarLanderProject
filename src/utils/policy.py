import random
import numpy as np

class EpsilonGreedyPolicy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    # Select action index based on epsilon-greedy policy
    def select_action(self, s, actions, Q):
        if random.random() < self.epsilon:
            return random.randint(0, len(actions) - 1)
        else:
            # I was having issues with the following line
            # return np.argmax([Q[(s,a)] for a in actions])
            values = [Q[(s,a)] for a in actions]
            max_value = np.max(values)
            max_indices = np.where(values == max_value)[0]
            a = np.random.choice(max_indices)
            return a
