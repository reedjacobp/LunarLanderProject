import random

class EpsilonGreedyPolicy:
    def __init__(self, epsilon):
        self.epsilon = epsilon

    # Select action index based on epsilon-greedy policy
    def select_action(self, state, q_values):
        if random.random() < self.epsilon:
            return random.randint(0, len(q_values) - 1)
        else:
            return max(range(len(q_values)), key=q_values.__getitem__)
