import os

this_dir = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import copy
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.policy import EpsilonGreedyPolicy
from gymnasium.utils.save_video import save_video

class QLearning:
    def __init__(self, env, n_episodes=1000, gamma=0.99, learning_rate=0.01, epsilon=0.1):
        self.env = env
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.policy = EpsilonGreedyPolicy()
        self.state_dict = {}
        self.step_starting_index = 0
        self.qlearning_data_path = os.path.abspath(os.path.join(this_dir, '..', '..', 'data', 'qlearning'))
        self.video_path = os.path.abspath(os.path.join(self.qlearning_data_path, 'videos'))
        self.plots_path = os.path.abspath(os.path.join(self.qlearning_data_path, 'plots'))

        # Initialize state dictionary
        # For the first 6 states, it will have 8 linearly spaced discretized points (num=8) 
        # and the last 2 states will have 2 linearly spaced discretized points
        sample_obs = env.observation_space.sample()
        for i, _ in enumerate(sample_obs):
            if i < 6:
                self.state_dict[i] = np.linspace(env.observation_space.low[i], env.observation_space.high[i], num=8)
            elif i >= 6:
                self.state_dict[i] = np.linspace(env.observation_space.low[i], env.observation_space.high[i], num=2)

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

    def create_q(self):
        """Helper function to create Q table"""
        print("Creating Q table...")
        discrete_sizes = [self.state_dict[s].size for s in range(self.env.observation_space.shape[0])]
        Q = {}
        for state_tuple in itertools.product(*[range(size) for size in discrete_sizes]):
            s = state_tuple
            for a in range(self.env.action_space.n):
                Q[(s,a)] = 0.0
        print("Finished creating Q table!")
        return Q

    def run_episode(self):

        Q = self.create_q()
        episodes = []
        with tqdm(total=self.n_episodes, desc='Running episodes', unit='episode') as pbar:
            for ep_index in range(self.n_episodes):
                episodes.append(self.episode(Q, self.env, ep_index, epsilon=max(self.epsilon, 1-ep_index/self.n_episodes)))
                pbar.update(1)
            return episodes

    def episode(self, Q, env, ep_index, epsilon, max_steps=200):

        # First pass
        obs = env.reset()
        s = self.discretize_state(obs[0])
        a = self.policy.select_action(s, list(range(env.action_space.n)), Q, epsilon)
        obs, r, terminated, truncated, _ = env.step(a)
        sp = self.discretize_state(obs)
        hist = [s]
        i = 0

        while not terminated and not truncated and i < max_steps:

            # Temporal Difference Update
            q_hat = r + self.gamma * np.max([Q[(sp, ap)] for ap in list(range(env.action_space.n))])

            # Learning Rate * Temporal Difference
            Q[(s, a)] += self.learning_rate * (q_hat - Q[(s, a)])

            # Stuff for next iteration
            s = sp
            a = self.policy.select_action(s, list(range(env.action_space.n)), Q, epsilon)
            obs, r, terminated, truncated, _ = env.step(a)
            sp = self.discretize_state(obs)
            hist.append(s)
            i += 1

            if terminated or truncated or i==max_steps:
                env_render = env.render()
                save_video(
                    env_render,
                    video_folder=self.video_path,
                    fps=env.metadata["render_fps"],
                    name_prefix="qlearning",
                    step_starting_index=self.step_starting_index,
                    episode_index=ep_index
                )
                self.step_starting_index += i
                i = 0
                if ep_index==self.n_episodes-1:
                    save_video(
                        env_render,
                        video_folder=self.video_path,
                        fps=self.env.metadata["render_fps"],
                        name_prefix="final_qlearning_video",
                    )

        return hist, Q
    
    def evaluate(self, env, policy, n_episodes_eval=1000, max_steps=200):
        
        # Limit the amount of episodes for evaluation
        if self.n_episodes > 1000:
            n_episodes_eval = 1000
        else:
            n_episodes_eval = self.n_episodes

        with tqdm(total=n_episodes_eval, desc='Evaluating', unit='episode') as pbar:
            returns = []
            for _ in range(n_episodes_eval):
                t = 0
                r = 0.0
                obs = env.reset()
                s = self.discretize_state(obs[0])
                terminated = False
                truncated = False
                while not terminated and not truncated and t < max_steps:
                    a = policy(s)
                    obs, r_act, terminated, truncated, _ = env.step(a)
                    r += r_act
                    s = self.discretize_state(obs)
                    t += 1
                returns.append(r)
                pbar.update()
        return returns
    
    def learn(self):
        episodes = self.run_episode()

        n = round(self.n_episodes/10)
        stop = self.n_episodes
        p1 = plt.subplot()
        p1.set_xlabel('steps in environment')
        p1.set_ylabel('avg return')

        print("Starting Q-table evaluation...")
        # Initialize Q, x-axis, and y-axis
        Q = self.create_q()
        xs = [0]
        ys = [np.mean([self.evaluate(self.env, lambda s: np.argmax([Q[(s, a)] for a in list(range(self.env.action_space.n))]))])]

        # Loop through each episode
        best_Q = None
        best_mean = float('-inf')
        with tqdm(total=(min(stop, len(episodes)) - n), desc='Evaluating Q-table') as pbar:
            for i in range(n, min(stop, len(episodes)), n):

                # This calculates how many steps were taken in this particular episode
                newsteps = sum(len(ep[0]) for ep in episodes[i-n:i])
                xs.append(xs[-1] + newsteps)

                # Grab the Q-table from this particular episode
                Q = episodes[i-n][1]

                # Compute the mean of undiscounted rewards from a simulation
                mean_reward = np.mean([self.evaluate(self.env, lambda s: np.argmax([Q[(s, a)] for a in list(range(self.env.action_space.n))]))])
                ys.append(mean_reward)

                # Keep track of the Q-table with the largest mean
                if mean_reward > best_mean:
                    best_Q = copy.deepcopy(Q)
                    best_mean = mean_reward

                # Update the progress bar
                pbar.update(n)

        # Plot the learning curve
        p1.plot(xs, ys, label='Q-Learning')
        plt.savefig(self.plots_path + '/learning_curve.png')

        # Save Q data
        np.save(self.qlearning_data_path + '/best_Q.npy', best_Q)

        # Print out information
        print(f'The best mean achieved was: {best_mean}')
        print(f'The best Q was saved to {self.qlearning_data_path}/best_Q.npy')
        print(f'Plots were saved to: {self.plots_path}')
        print(f'Videos were saved to: {self.video_path}')
        