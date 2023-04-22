import gymnasium as gym
from algorithms.q_learning import QLearning
from algorithms.dqn import DQNAgent

def main():
    env = gym.make("LunarLander-v2", render_mode="rgb_array_list")
    
    algorithm = input("Enter algorithm to use (q, dq, sarsa, dqn): ")

    if algorithm == 'q':
        q = QLearning(env, n_episodes=10000)
        q.learn()
    # elif algorithm == 'dq':
    #     dq = DoubleQLearning(env, n_episodes=10000)
    #     dq.learn()
    # elif algorithm == 'sarsa':
    #     sarsa = SARSA(env, n_episodes=10000)
    #     sarsa.learn()
    elif algorithm == 'dqn':
        dqn_agent = DQNAgent(env, n_episodes=500)
        dqn_agent.train()
    else:
        print("Invalid algorithm selected.")

    env.close()

if __name__ == '__main__':
    main()
    