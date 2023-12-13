"""
Julia Santaniello
06/25/23

For training a single policy. Input hyperparamters and saved policy name.
"""

#import pygame
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time

class Train():
    def __init__(self, environment, env_name, algorithm, episodes, steps, gamma, alpha, seed, name):
        self.env_name = env_name
        self.steps = steps
        self.env = environment
        self.policy = algorithm
        self.eps = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.name = name
        self.seed = seed

    def run(self):
        print("Starting agent training...")
        rewards, steps = self.train()
        self.save_data(rewards, steps)
        print("Done!")

    def save_data(self, rewards, steps):
        print("passed saved")
        pass
        
    def train(self):
        avg_rewards_past = [] 
        avg_steps_past = []

        for i in range(self.eps):
            print("ep: ", i)
            steps = self.steps
            state = self.env.reset()

            running_reward, running_steps = 0

            for j in range(steps):
                exploring = False

                #pick an action
                rand = np.random.uniform(0,1)
                if rand < self.epsilon: exploring == True
                action = self.policy.process_step(state, exploring)
                next_state, reward, done, win= self.env.step(action)

                self.policy.save_rewards(reward,state,action,next_state,done)
                running_reward += reward
                state = next_state

                if done:
                    running_steps += j
                    break
            
                self.policy.finish_episode(running_reward, False)

            if i % 20 == 0:
                average_reward = running_reward/20
                average_step = running_steps/20
                avg_rewards_past.append(average_reward)
                avg_steps_past.append(average_step)
                print('Episode {}\tlength: {}\treward: {}'.format(i, j, average_reward))
                running_reward, running_steps = 0

            self.policy.update_parameters()
        
        self.policy.save_agent()

        return avg_rewards_past, avg_steps_past

def plots(rewards_orig, rewards_mod):
    iterations = range(0, len(rewards_orig), 1)
    #print(iterations)
    #print(rewards_orig)
    plt.plot(iterations, rewards_orig)
    plt.plot(iterations, rewards_mod)
    plt.ylabel("Average Return")
    plt.xlabel("Iterations")
    plt.legend(["Original", "Modified"])
    plt.show()
        

if __name__ == "__main__":
    # rewards_orig = train(0.99, 0.01, 2000, 800, "MODIFIED_JS1", False) #Gamma, lr, episodes, steps, path name
    # rewards_mod = train(0.99, 0.01, 2000, 800, "ORIGINAL_JS1", True) #Gamma, lr, episodes, steps, path name
    # plots(rewards_orig, rewards_mod)
    pass
