import gymnasium as gym
from algorithms.q_learning import QLearning

def main():
    env = gym.make("LunarLander-v2", render_mode="rgb_array_list")
    
    # Do Q Learning
    q = QLearning(env, n_episodes=50001)
    q.learn()

    # Do Double Q Learning

    # Do SARSA

    # Do DQN

if __name__ == '__main__':
    main()
    