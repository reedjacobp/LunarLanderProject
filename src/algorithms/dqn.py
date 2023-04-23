import os

this_dir = os.path.dirname(os.path.abspath(__file__))

import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from gymnasium.utils.save_video import save_video
from utils.plotting import plot_learning_curve, error_plots

class DQN(nn.Module):
    def __init__(self, env, lr, fc1_dims, fc2_dims, fc3_dims):
        super(DQN,self).__init__()
        self.input_dims = [env.observation_space.shape[0]]
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc3_dims
        self.n_actions = env.action_space.n
        self.fc1 = nn.Linear(*self.input_dims,self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims,self.fc3_dims)
        self.fc4 = nn.Linear(self.fc3_dims,self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
        self.loss_calc = 0.0

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.fc4(x)
        return actions

class DQNAgent():
    def __init__(self, env, n_episodes=100, gamma=0.99, epsilon=1.0, lr=1e-3, batch_size=100, mmem_size=100000, max_steps=700, eps_end=0.01, eps_dec=5e-4):
        # Setup agent
        self.env = env
        self.n_episodes = n_episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.action_space = list(range(env.action_space.n))
        self.mem_size = mmem_size
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.mem_cnter = 0
        self.Q_eval = DQN(env, lr, fc1_dims=200, fc2_dims=200, fc3_dims=200)

        #build memory holders
        self.state_memory = np.zeros((self.mem_size, *[env.observation_space.shape[0]]), dtype=np.float32) # build memory holder
        self.new_state_memory = np.zeros((self.mem_size, *[env.observation_space.shape[0]]), dtype=np.float32) # build memory holder
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory =np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool_)

        # Stuff for data, plots, and videos
        self.step_starting_index = 0
        self.dqn_data_path = os.path.abspath(os.path.join(this_dir, '..', '..', 'data', 'dqn'))
        self.video_path = os.path.abspath(os.path.join(self.dqn_data_path, 'videos'))
        self.plots_path = os.path.abspath(os.path.join(self.dqn_data_path, 'plots'))

    # Store experience into buffer
    def store_transition(self,state,action,reward,state_,terminated):
        index = self.mem_cnter % self.mem_size # can overwrite oldest memories
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index]=  reward
        self.terminal_memory[index] = terminated
        self.action_memory[index] = action
        self.mem_cnter += 1

    # Epsilon-Greedy Policy
    def select_action(self,observation):
        if np.random.random() > self.epsilon:
            state = T.Tensor(np.array(observation)).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item() # convert from tensor
        else:
            action = np.random.choice(self.action_space)
        return action 

    def learn(self):

        # Just explore until the batch is full
        if self.mem_cnter < self.batch_size:
            return 0.0
        
        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cnter, self.mem_size) # take the less full one
        batch = np.random.choice(max_mem, self.batch_size, replace=False) # get rid of duplicate memories for learning

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.Q_eval.device) # turn a subset of memory into a pytorch tensor
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        action_batch = self.action_memory[batch]

        # drive agent to max action
        q_eval = self.Q_eval.forward(state_batch)[batch_index,action_batch] # grab actions you took 
        q_next = self.Q_eval.forward(new_state_batch)# get next actions
        q_next[terminal_batch] = 0.0
        q_target = reward_batch+self.gamma*T.max(q_next,dim=1)[0] # grab value not index

        loss = self.Q_eval.loss(q_target,q_eval).to(self.Q_eval.device)
        loss.backward() # back propagate

        self.Q_eval.optimizer.step()

        # decay epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min \
            else self.eps_min
        
        return loss.item()

    def train(self, msim=None):
        # Initialize metrics
        scores, losses, stds, avg_scores, avg_losses, eps_history = [], [], [], [], [], []
        best_score = float('-inf')

        # Loop over the range of the number of episodes
        for ep_index in range(self.n_episodes):
            # Initialize the experience for this episode
            score = 0
            terminated = False
            truncated = False
            obs = self.env.reset()
            s = obs[0]
            env_step = 0

            while not terminated and not truncated and env_step < self.max_steps:
                action = self.select_action(s)
                
                # Gather experience
                sp, r, terminated, truncated, _ = self.env.step(action)

                # Increment score
                score += r
                
                # Store experience into buffer
                self.store_transition(s, action, r, sp, terminated)
                loss = self.learn()

                # Setup for next iteration
                s = sp 
                env_step += 1

                # Save environment rendering
                if (terminated or truncated or env_step==self.max_steps) and msim is None:
                    env_render = self.env.render()
                    save_video(
                        env_render,
                        video_folder=self.video_path,
                        fps=self.env.metadata["render_fps"],
                        name_prefix="dqn_video",
                        step_starting_index=self.step_starting_index,
                        episode_index=ep_index
                    )
                    self.step_starting_index += env_step
                    env_step = 0
                    if ep_index==self.n_episodes-1:
                        save_video(
                            env_render,
                            video_folder=self.video_path,
                            fps=self.env.metadata["render_fps"],
                            name_prefix="final_dqn_video",
                        )

            if score > best_score and msim is None:
                best_score = score
                if best_score > 100.0:
                    save_video(
                        env_render,
                        video_folder=self.video_path,
                        name_prefix=f'best_dqn_score{best_score:.2f}',
                        fps=self.env.metadata["render_fps"],
                    )

            # Store the score for this episode
            scores.append(score)
            losses.append(loss)
            eps_history.append(self.epsilon)

            # Compute average score and std over the last 25 episodes
            avg_loss = np.mean(losses[-25:])
            avg_score = np.mean(scores[-25:])
            avg_scores.append(avg_score)
            avg_losses.append(avg_loss)
            std = np.std(scores[-25:])
            stds.append(std)

            print('episode', ep_index, 'score %.2f' % score,'avg score %.2f' % avg_score)

        if msim is not None:
            plot_learning_curve(list(range(1, self.n_episodes+1)), losses, avg_losses, self.plots_path + f'/loss_learning_curve_mc_{msim+1}.png', 'loss')
            plot_learning_curve(list(range(1, self.n_episodes+1)), scores, avg_scores, self.plots_path + f'/score_learning_curve_mc_{msim+1}.png', 'score')
        else:
            plot_learning_curve(list(range(1, self.n_episodes+1)), losses, avg_losses, self.plots_path + f'/loss_learning_curve.png', 'loss')
            plot_learning_curve(list(range(1, self.n_episodes+1)), scores, avg_scores, self.plots_path + f'/score_learning_curve.png', 'score')

        return scores, avg_score, std
    
    def run_mc(self, msims=10):
        mc_avg_scores, mc_stds = [], []
        for msim in range(msims):
            print(f'\n****** MC Run {msim+1} ******')
            _, avg_score, std = self.train(msim=msim)
            mc_avg_scores.append(avg_score)
            mc_stds.append(std)

        error_plots(list(range(1, msims+1)), mc_avg_scores, mc_stds, self.plots_path + f'/err_mc.png')
