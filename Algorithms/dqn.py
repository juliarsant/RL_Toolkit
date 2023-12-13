# import random
# import copy
# import gym
# from gym.core import Env
# import torch
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# from torch import nn as nn
# from torch.optim import AdamW
# from tqdm import tqdm
# import numpy as np


# class PreprocessEnv(gym.Wrapper):
#     def __init__(self, env: Env):
#         gym.Wrapper.__init__(self, env)

#     def reset(self):
#         obs, _ = self.self.env.reset()
#         return torch.from_numpy(obs).unsqueeze(dim=0).float()
    
#     def step(self, action):
#         action = action.item()
#         next_state, reward, done, trunc, info = self.self.env.step(action)
#         next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
#         reward = torch.tensor(reward).view(1, -1).float()
#         done = torch.tensor(done).view(1, -1)
#         return next_state, reward, done, info
    


# class ReplayMemory:
#     def __init__(self, capacity = 100000):
#         self.capacity = capacity
#         self.memory = []
#         self.position = 0

#     def insert(self, transition):
#         if len(self.memory) < self.capacity:
#             self.memory.append(None)
#         self.memory[self.position] = transition
#         self.position = (self.position + 1) % self.capacity

#     def sample(self):
#         assert self.can_sample(self.batch_size)
#         item_list = []

#         batch = random.sample(self.memory, self.batch_size)

#         batch = list(zip(*batch))

#         return [torch.cat(items) for items in batch]
    
#     def can_sample(self, batch_size):
#         return len(self.memory) >= batch_size * 10
    
#     def __len__(self):
#         return len(self.memory)

# class DQN():
#     def __init__(self, env, env_name, demo_name, num_actions, obs_space, obs_space_type, episodes, steps=None, alpha = 0.0001, batch_size = 32, gamma = 0.99, epsilon = 0.2) -> None:

#         self.state_dims = obs_space
#         self.num_actions = num_actions
#         self.env_name = env_name
#         self.demo_name = demo_name


#         self.env = PreprocessEnv(env)
#         self.batch_size = batch_size
#         self.episodes = episodes
#         self.steps = steps
#         self.gamma = gamma
#         self.epsilon = epsilon

#         # some temp variables
#         self._xs,self._a,self._drs = [],[],[]

#         # some hitl temp variables
#         self.h_actions, self.h_rewards = [], []
#         self.a_probs = []

#         self.init_model()

#     def set_explore_epsilon(self, epsilon):
#         self.epsilon = epsilon

#     #Saves an array that represents the human demonstration "distribution"
#     def save_human_action(self, human_action):
#         x = np.zeros(self.num_actions)
#         x[human_action] = 1.0
        
#         self.h_actions.append(x)

#     def init_model(self):
#         self.q_network = nn.Sequential(
#             nn.Linear(self.state_dims, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, self.num_actions)
#         )
#         self.memory = ReplayMemory()
#         self.target_self = copy.deepcopy(self.q_network).eval()
#         self.optim = AdamW(self.q_network.parameters(), lr = self.alpha)
#         self.stats = {'MSE Loss': [], 'Returns': []}

#     def policy(self, state, exploring):
#         if exploring:
#             return torch.randint(self.num_actions, (1,1))
#         else:
#             av = self.q_network(state[0].clone().detach())
#             return torch.unsqueeze(torch.argmax(av, dim=-1, keepdim=True),0)
    
#     def save_agent(self):
#         torch.save(policy.q_network(), './Trained_Agents/{}/{}.pth'.format(self.env_name, self.demo_name))

#     def process_step(self, state, exploring=None):
#         action = policy(state, exploring)
# 		# record various intermediates (needed later for backprop)
#         self._xs.append(state) # observation
#         self._a.append(action)

#         return action
    
#     def save_rewards(self, reward, state,action,next_state,done):
#         # store the reward in the list of rewards
#         self._drs.append(reward)
#         self.memory.insert([state, action, reward, done, next_state])

#         if self.memory.can_sample(self.batch_size):
#             state_b, action_b, reward_b, _, next_state_b = self.memory.sample(self.batch_size)
#             qsa_b = self.q_network(state_b).gather(1, action_b)

#             next_qsa_b = self.target_self(next_state_b)
#             next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]

#             target_b = reward_b + self.gamma * next_qsa_b
#             loss = F.mse_loss(qsa_b, target_b)

#             self.q_network.zero_grad()
#             loss.backward()
#             self.optim.step()

#             self.stats['MSE Loss'].append(loss)

#     def finish_episode(self, ep_return, human_demonstration=False):
#         self.stats['Returns'].append(ep_return)


#     def update_parameters(self):
#         self.q_network.load_state_dict(self.q_network.state_dict())

#     def deep_q_learning(self):
#         state = self.env.reset()
#         action = torch.tensor(0)
#         next_state, reward, done, _ = self.env.step(action)

#         optim = AdamW(self.q_network.parameters(), lr = self.alpha)
#         memory = ReplayMemory()
#         stats = {'MSE Loss': [], 'Returns': []}

#         for episode in tqdm(range(1, self.episodes+1)):
#             state = self.env.reset()
#             done = False
#             ep_return = 0

#             for self.steps in range(100):
#                 action = policy(state, self.epsilon)
#                 next_state, reward, done, _ = self.env.step(action)
#                 memory.insert([state, action, reward, done, next_state])

#                 if memory.can_sample(self.batch_size):
#                     state_b, action_b, reward_b, _, next_state_b = memory.sample(self.batch_size)
#                     qsa_b = self.q_network(state_b).gather(1, action_b)

#                     next_qsa_b = self.target_self(next_state_b)
#                     next_qsa_b = torch.max(next_qsa_b, dim=-1, keepdim=True)[0]

#                     target_b = reward_b + self.gamma * next_qsa_b
#                     loss = F.mse_loss(qsa_b, target_b)

#                     self.q_network.zero_grad()
#                     loss.backward()
#                     optim.step()

#                     stats['MSE Loss'].append(loss)
#                 state = next_state
#                 ep_return += reward.item()
            
#             stats['Returns'].append(ep_return)
        
#             if episode % 10 == 0:
#                 self.q_network.load_state_dict(self.q_network.state_dict())

#         return stats

# """

# This module contains wrappers and convenience functions to simplify
# working with gym environments of different kinds.

# """
# from typing import Callable

# from IPython import display
# import matplotlib.patches as mpatches
# import matplotlib.pyplot as plt
# import seaborn as sns
# import gym
# import torch
# import numpy as np


# def plot_policy(probs_or_qvals, frame, action_meanings=None):
#     if action_meanings is None:
#         action_meanings = {0: 'U', 1: 'R', 2: 'D', 3: 'L'}
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     max_prob_actions = probs_or_qvals.argmax(axis=-1)
#     probs_copy = max_prob_actions.copy().astype(np.object)
#     for key in action_meanings:
#         probs_copy[probs_copy == key] = action_meanings[key]
#     sns.heatmap(max_prob_actions, annot=probs_copy, fmt='', cbar=False, cmap='coolwarm',
#                 annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])
#     axes[1].imshow(frame)
#     axes[0].axis('off')
#     axes[1].axis('off')
#     plt.suptitle("Policy", size=24)
#     plt.tight_layout()


# def plot_values(state_values, frame):
#     f, axes = plt.subplots(1, 2, figsize=(12, 5))
#     sns.heatmap(state_values, annot=True, fmt=".2f", cmap='coolwarm',
#                 annot_kws={'weight': 'bold', 'size': 12}, linewidths=2, ax=axes[0])
#     axes[1].imshow(frame)
#     axes[0].axis('off')
#     axes[1].axis('off')
#     plt.tight_layout()


# def plot_action_values(action_values):

#     text_positions = [
#         [(0.35, 4.75), (1.35, 4.75), (2.35, 4.75), (3.35, 4.75), (4.35, 4.75),
#          (0.35, 3.75), (1.35, 3.75), (2.35, 3.75), (3.35, 3.75), (4.35, 3.75),
#          (0.35, 2.75), (1.35, 2.75), (2.35, 2.75), (3.35, 2.75), (4.35, 2.75),
#          (0.35, 1.75), (1.35, 1.75), (2.35, 1.75), (3.35, 1.75), (4.35, 1.75),
#          (0.35, 0.75), (1.35, 0.75), (2.35, 0.75), (3.35, 0.75), (4.35, 0.75)],
#         [(0.6, 4.45), (1.6, 4.45), (2.6, 4.45), (3.6, 4.45), (4.6, 4.45),
#          (0.6, 3.45), (1.6, 3.45), (2.6, 3.45), (3.6, 3.45), (4.6, 3.45),
#          (0.6, 2.45), (1.6, 2.45), (2.6, 2.45), (3.6, 2.45), (4.6, 2.45),
#          (0.6, 1.45), (1.6, 1.45), (2.6, 1.45), (3.6, 1.45), (4.6, 1.45),
#          (0.6, 0.45), (1.6, 0.45), (2.6, 0.45), (3.6, 0.45), (4.6, 0.45)],
#         [(0.35, 4.15), (1.35, 4.15), (2.35, 4.15), (3.35, 4.15), (4.35, 4.15),
#          (0.35, 3.15), (1.35, 3.15), (2.35, 3.15), (3.35, 3.15), (4.35, 3.15),
#          (0.35, 2.15), (1.35, 2.15), (2.35, 2.15), (3.35, 2.15), (4.35, 2.15),
#          (0.35, 1.15), (1.35, 1.15), (2.35, 1.15), (3.35, 1.15), (4.35, 1.15),
#          (0.35, 0.15), (1.35, 0.15), (2.35, 0.15), (3.35, 0.15), (4.35, 0.15)],
#         [(0.05, 4.45), (1.05, 4.45), (2.05, 4.45), (3.05, 4.45), (4.05, 4.45),
#          (0.05, 3.45), (1.05, 3.45), (2.05, 3.45), (3.05, 3.45), (4.05, 3.45),
#          (0.05, 2.45), (1.05, 2.45), (2.05, 2.45), (3.05, 2.45), (4.05, 2.45),
#          (0.05, 1.45), (1.05, 1.45), (2.05, 1.45), (3.05, 1.45), (4.05, 1.45),
#          (0.05, 0.45), (1.05, 0.45), (2.05, 0.45), (3.05, 0.45), (4.05, 0.45)]]

#     fig, ax = plt.subplots(figsize=(9, 9))
#     tripcolor = quatromatrix(action_values, ax=ax,
#                              triplotkw={"color": "k", "lw": 1}, tripcolorkw={"cmap": "coolwarm"})
#     ax.margins(0)
#     ax.set_aspect("equal")
#     fig.colorbar(tripcolor)

#     for j, av in enumerate(text_positions):
#         for i, (xi, yi) in enumerate(av):
#             plt.text(xi, yi, round(action_values[:, :, j].flatten()[i], 2), size=10, color="w", weight="bold")

#     plt.title("Action values Q(s,a)", size=18)
#     plt.tight_layout()
#     plt.show()


# def quatromatrix(action_values, ax=None, triplotkw=None, tripcolorkw=None):
#     action_values = np.flipud(action_values)
#     n = 5
#     m = 5
#     a = np.array([[0, 0], [0, 1], [.5, .5], [1, 0], [1, 1]])
#     tr = np.array([[0, 1, 2], [0, 2, 3], [2, 3, 4], [1, 2, 4]])
#     A = np.zeros((n * m * 5, 2))
#     Tr = np.zeros((n * m * 4, 3))
#     for i in range(n):
#         for j in range(m):
#             k = i * m + j
#             A[k * 5:(k + 1) * 5, :] = np.c_[a[:, 0] + j, a[:, 1] + i]
#             Tr[k * 4:(k + 1) * 4, :] = tr + k * 5
#     C = np.c_[action_values[:, :, 3].flatten(), action_values[:, :, 2].flatten(),
#               action_values[:, :, 1].flatten(), action_values[:, :, 0].flatten()].flatten()

#     ax.triplot(A[:, 0], A[:, 1], Tr, **triplotkw)
#     tripcolor = ax.tripcolor(A[:, 0], A[:, 1], Tr, facecolors=C, **tripcolorkw)
#     return tripcolor


# def test_agent(env: gym.Env, policy: Callable, episodes: int = 10) -> None:
#     plt.figure(figsize=(8, 8))
#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         img = plt.imshow(env.render(mode='rgb_array'))
#         while not done:
#             p = policy(state)
#             if isinstance(p, np.ndarray):
#                 action = np.random.choice(4, p=p)
#             else:
#                 action = p
#             next_state, _, done, _ = env.step(action)
#             img.set_data(env.render(mode='rgb_array'))
#             plt.axis('off')
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#             state = next_state


# def plot_cost_to_go(env, q_network, xlabel=None, ylabel=None):
#     highx, highy = env.observation_space.high
#     lowx, lowy = env.observation_space.low
#     X = torch.linspace(lowx, highx, 100)
#     Y = torch.linspace(lowy, highy, 100)
#     X, Y = torch.meshgrid(X, Y)

#     q_net_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
#     Z = - q_network(q_net_input).max(dim=-1, keepdim=True)[0]
#     Z = Z.reshape(100, 100).detach().numpy()
#     X = X.numpy()
#     Y = Y.numpy()

#     fig = plt.figure(figsize=(8, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(X, Y, Z, cmap='jet', linewidth=0, antialiased=False)
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     ax.set_xlabel(xlabel, size=14)
#     ax.set_ylabel(ylabel, size=14)
#     ax.set_title("Estimated cost-to-go", size=18)
#     plt.tight_layout()
#     plt.show()


# def plot_tabular_cost_to_go(action_values, xlabel, ylabel):
#     plt.figure(figsize=(8, 8))
#     cost_to_go = -action_values.max(axis=-1)
#     plt.imshow(cost_to_go, cmap='jet')
#     plt.title("Estimated cost-to-go", size=24)
#     plt.xlabel(xlabel, size=18)
#     plt.ylabel(ylabel, size=18)
#     plt.xticks([])
#     plt.yticks([])
#     plt.xticks()
#     plt.colorbar()
#     plt.tight_layout()
#     plt.show()


# def plot_max_q(env, q_network, xlabel=None, ylabel=None, action_labels=[]):
#     highx, highy = env.observation_space.high
#     lowx, lowy = env.observation_space.low
#     X = torch.linspace(lowx, highx, 100)
#     Y = torch.linspace(lowy, highy, 100)
#     X, Y = torch.meshgrid(X, Y)
#     q_net_input = torch.stack([X.flatten(), Y.flatten()], dim=-1)
#     Z = q_network(q_net_input).argmax(dim=-1, keepdim=True)
#     Z = Z.reshape(100, 100).T.detach().numpy()
#     values = np.unique(Z.ravel())
#     values.sort()

#     plt.figure(figsize=(5, 5))
#     plt.xlabel(xlabel, size=14)
#     plt.ylabel(ylabel, size=14)
#     plt.title("Optimal action", size=18)

#     # im = plt.imshow(Z, interpolation='none', cmap='jet')
#     im = plt.imshow(Z, cmap='jet')
#     colors = [im.cmap(im.norm(value)) for value in values]
#     patches = [mpatches.Patch(color=color, label=label) for color, label in zip(colors, action_labels)]
#     plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
#     plt.tight_layout()


# def plot_stats(stats):
#     rows = len(stats)
#     cols = 1

#     fig, ax = plt.subplots(rows, cols, figsize=(12, 6))

#     for i, key in enumerate(stats):
#         vals = stats[key]
#         vals = [np.mean(vals[i-10:i+10]) for i in range(10, len(vals)-10)]
#         if len(stats) > 1:
#             ax[i].plot(range(len(vals)), vals)
#             ax[i].set_title(key, size=18)
#         else:
#             ax.plot(range(len(vals)), vals)
#             ax.set_title(key, size=18)
#     plt.tight_layout()
#     plt.show()


# def seed_everything(env: gym.Env, seed: int = 42) -> None:
#     """
#     Seeds all the sources of randomness so that experiments are reproducible.
#     Args:
#         env: the environment to be seeded.
#         seed: an integer seed.
#     Returns:
#         None.
#     """
#     env.seed(seed)
#     env.action_space.seed(seed)
#     env.observation_space.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.set_deterministic(True)


# def test_policy_network(env, policy, episodes=1):
#     from IPython import display
#     plt.figure(figsize=(6, 6))
#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         img = plt.imshow(env.render(mode='rgb_array'))
#         while not done:
#             state = torch.from_numpy(state).unsqueeze(0).float()
#             action = policy(state).multinomial(1).item()
#             next_state, _, done, _ = env.step(action)
#             img.set_data(env.render(mode='rgb_array'))
#             plt.axis('off')
#             display.display(plt.gcf())
#             display.clear_output(wait=True)
#             state = next_state


# def plot_action_probs(probs, labels):
#     plt.figure(figsize=(6, 4))
#     plt.bar(labels, probs, color ='orange')
#     plt.title("$\pi(s)$", size=16)
#     plt.xticks(fontsize=12)
#     plt.yticks(fontsize=12)
#     plt.tight_layout()
#     plt.show()

# if __name__ == "__main__":
#     policy = DQN(None)
#     # stats = policy.deep_q_learning(self.q_network, policy, 500)
#     # print(stats)
#     # plot_stats(stats)
#     # test_agent(env, policy, epsiodes = 2)


