# from gym_envs.lunar_lander import LunarLander
import gymnasium as gym
from algorithms.q_learning import QLearning
import numpy as np
import matplotlib.pyplot as plt

def main():
    env = gym.make("LunarLander-v2")
    # env.seed(0)
    # np.random.seed(0)
    
    # Run your training or testing loop here
    q = QLearning(env)
    episodes = q.run_episode(env, n_episodes=100)

    n = 10
    stop = 100
    p1 = plt.subplot()
    p1.set_xlabel('steps in environment')
    p1.set_ylabel('avg return')
    for eps in episodes:
        Q = np.zeros((env.observation_space.shape[0], env.action_space.n))
        xs = [0]
        # calc_returns = q.evaluate(env, lambda s: np.argmax([Q[(s, a)] for a in list(range(env.action_space.n))]))
        ys = [np.mean(q.evaluate(env, Q))]
        for i in range(n, min(stop, len(eps)), n):
            newsteps = sum(len(ep.hist) for ep in eps[i-n+1:i])
            xs.append(xs[-1] + newsteps)
            Q = eps[i].Q
            ys.append(np.mean([q.evaluate(env, lambda s: np.argmax([Q[(s, a)] for a in list(range(env.action_space.n))]))]))
        p1.plot(xs, ys, label='Q-Learning')
    plt.savefig('steps_in_env.png')

if __name__ == '__main__':
    main()
    