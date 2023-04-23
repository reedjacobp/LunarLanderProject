import gymnasium as gym
from algorithms.q_learning import QLearning
from algorithms.dqn import DQNAgent

def main():
    env = gym.make("LunarLander-v2", render_mode="rgb_array_list")
    
    algorithm = input("Enter algorithm to use (q, dq, sarsa, dqn): ")

    if algorithm == 'q':
        q = QLearning(env, n_episodes=10000)
        q.learn()
    elif algorithm == 'dqn':
        # Iteration 1 should be using the relu activation function
        # dqn_agent_iter1 = DQNAgent(env, n_episodes=500, lr=1e-4)
        # dqn_agent_iter1.train()

        # Iteration 2 should be using the relu activation function
        # dqn_agent_iter2 = DQNAgent(env, n_episodes=400, lr=1e-4, batch_size=1000, mmem_size=1000000, max_steps=500, eps_dec=1e-4)
        # dqn_agent_iter2.train()

        # Iteration 3 should be using the tanh activation function
        dqn_agent_iter3 = DQNAgent(env, n_episodes=50, lr=1e-4, batch_size=5, mmem_size=1000000, max_steps=500, eps_dec=1e-4)
        dqn_agent_iter3.train()

        # TODO: Add some logic to determine if the user wants to run a MC
        # dqn_agent.run_mc(msims=10)
    else:
        print("Invalid algorithm selected!")

    env.close()

if __name__ == '__main__':
    main()
    