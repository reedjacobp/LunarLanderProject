from utils.policy import EpsilonGreedyPolicy
import numpy as np

class QLearning:
    def __init__(self, env, gamma=0.99, learning_rate=0.01, epsilon=0.10):
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.policy = EpsilonGreedyPolicy(epsilon)
        self.state_dict = {}
        self.Q = None

        # Initialize state dictionary
        sample_obs = env.observation_space.sample()
        for i, _ in enumerate(sample_obs):
            self.state_dict[i] = np.linspace(env.observation_space.low[i], env.observation_space.high[i], num=env.observation_space.shape[0])

    def discretize_state(self, state):
        """Discretize continuous state to nearest discrete state."""
        disc_state = []
        for i in range(len(self.state_dict)):
            if i < len(state):
                if (state[i] <= self.state_dict[i][0]).all():
                    disc_state.append(0)
                elif (state[i] >= self.state_dict[i][-1]).all():
                    disc_state.append(len(self.state_dict[i]) - 1)
                else:
                    disc_state.append(np.argmin(np.abs(self.state_dict[i] - state[i])))
            else:
                disc_state.append(0)
        return tuple(disc_state)

    def run_episode(self, env, n_episodes):
        self.Q = np.zeros((env.observation_space.shape[0], env.action_space.n))
        episodes = []

        for _ in range(n_episodes):
            obs = env.reset()
            s = self.discretize_state(obs[0])
            episodes.append(self.episode(s, env))

        return episodes


    def episode(self, s, env):
        terminated = False
        truncated = False
        hist = [s]

        for _ in range(1000):
            # Choose action
            a = self.policy.select_action(s, list(range(env.action_space.n)), self.Q)

            # Take action
            obs, r, terminated, truncated, info = env.step(a)
            sp = self.discretize_state(obs)

            if terminated or truncated:
                obs = env.reset()
                s = self.discretize_state(obs[0])
            else:
                # Update Q value
                q_hat = r + self.gamma * np.max(self.Q[(sp, a)])
                self.Q[(sp, a)] += self.learning_rate * (q_hat - self.Q[(sp, a)])

                # Update state and history
                s = sp
                hist.append(s)

        return hist, self.Q
