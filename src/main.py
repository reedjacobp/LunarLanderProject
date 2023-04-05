# from gym_envs.lunar_lander import LunarLander
import gymnasium as gym
import numpy as np
from algorithms.q_learning import QLearning

def main():
    env = gym.make("LunarLander-v2", render_mode="human")
    # env.seed(0)
    # np.random.seed(0)
    
    # Run your training or testing loop here
    q = QLearning(env)
    q.run_episode(env, n_episodes=1_200_000)

if __name__ == '__main__':
    main()
    