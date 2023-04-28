import gymnasium as gym
from algorithms.q_learning import QLearning
from algorithms.dqn import DQNAgent
import numpy as np
from utils.plotting import error_plots
import time

def main():
    env = gym.make("LunarLander-v2", render_mode="rgb_array_list")
    
    algorithm = input("Enter algorithm to use (q, dqn): ")

    if algorithm == 'q':
        q = QLearning(env, n_episodes=10000)
        q.learn()
    elif algorithm == 'dqn':
        # Iteration 1 should be using the relu activation function
        # dqn_agent_iter1 = DQNAgent(env, n_episodes=500, lr=1e-4)
        # dqn_agent_iter1.train()

        # Iteration 2 should be using the relu activation function and 3 layer NN with 200 neurons each
        # dqn_agent_iter2 = DQNAgent(env, n_episodes=400, lr=1e-4, batch_size=1000, buffer_size=1000000, max_steps=500, eps_dec=1e-4)
        # dqn_agent_iter2.train()

        # Iteration 3 should be using the relu activation function and 2 layer NN with 128 neurons each
        # dqn_agent_iter3 = DQNAgent(env, n_episodes=400, lr=5e-4, batch_size=64, buffer_size=1000000, max_steps=1000, eps_dec=1e-4)
        # dqn_agent_iter3.train()

        # Change the dqn_agent_iter# to whatever object (dqn_agent_iter1/2/3) you want to run the MC sims for
        msims = 10
        scores_hist, time_hist, mc_avg_scores, mc_stds = [], [], [], []
        for msim in range(msims):
            start = time.time()
            print(f'\n****** MC Run {msim+1} ******')

            dqn_agent = DQNAgent(env, n_episodes=400, lr=1e-4, batch_size=1000, buffer_size=1000000, max_steps=500, eps_dec=1e-4)

            scores, avg_score, std = dqn_agent.train(msim=msim)
            scores_hist.append(scores)
            time_hist.append(time.time()-start)
            mc_avg_scores.append(avg_score)
            mc_stds.append(std)

            print(f'Time to complete MC Run {msim+1}: {time_hist[msim]}')

        error_plots(list(range(1, msims+1)), mc_avg_scores, mc_stds, dqn_agent.plots_path + f'/err_mc.png')
        mean_mc = np.mean(scores_hist)
        sem_mc = np.std(scores_hist)/np.sqrt(msims)
        mean_time_mc = np.mean(time_hist)
        print(f'\nMean Score over the MC simulations: {mean_mc}')
        print(f'Standard Error of the Mean over the MC simulations: {sem_mc}')
        print(f'Average time to complete each MC simulation: {mean_time_mc}')
        


    else:
        print("Invalid algorithm selected!")

    env.close()

if __name__ == '__main__':
    main()
    