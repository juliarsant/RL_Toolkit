"""
Julia Santaniello
06/25/23

For training a single policy. Input hyperparamters and saved policy name.
"""

#import pygame
import time
import pygame
from simplePG import SimplePG
import torch
import torch.optim as optim
from lunar_lander import LunarLander 
import random
import matplotlib.pyplot as plt
import gym

def human_play():
    pygame.init()
    pressed_keys = pygame.key.get_pressed()

    if pressed_keys[pygame.K_LEFT]: #left
        return 1
    elif pressed_keys[pygame.K_UP]: #up
        return 2
    elif pressed_keys[pygame.K_RIGHT]: #right
        return 3
    
    return 0 #down

def train(gamma, lr, eps, steps, title, version):
    
    render = False #rendering
    epsilon = 0.1
    D = 8 * 2 * 11 + 2
    if version:
        env = LunarLander(render_mode="human") #modified LunarLander
        policy = SimplePG(4, 11, 11, lr, gamma, 0.99, 0.1, random_seed=10) #11 state elements in modified LunarLander

    else:
        env = gym.make('LunarLander-v2')
        policy = SimplePG(4, 8, 8, lr, gamma, 0.99, 0.1, random_seed=10) #8 state elements in original LunarLander
    
    policy.set_explore_epsilon(0.1)
    
    rewards_past = []
    avg_rewards_past = []
    running_reward = 0


    for i_episode in range(0, eps):
        episode_rewards = 0

        state, _ = env.reset()

        for t in range(steps):

            action = policy.process_step(state, True)
            human_action = human_play()
            policy.save_human_action(human_action)

            state, reward, done, _, win = env.step(action)
            policy.give_reward(reward)
            running_reward += reward
            episode_rewards += reward

            if render and (i_episode > 500 and i_episode%20 == 0):
                env.render()
            if done:
                break
        
        rewards_past.append(episode_rewards)
        if i_episode % 20 == 0:
            average_reward = running_reward/20
            avg_rewards_past.append(average_reward)
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, average_reward))
            running_reward = 0

        
        # saving the model if episodes > 999 OR avg reward > 200 
        #if i_episode > eps-1:
        #    torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(title))
        
        # if average_reward > 210:
        #     torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(title))
        #     print("########## Solved! ##########")
        #     #test(name='LunarLander_{}}.pth')#.format(title))
        #     break
    
        policy.finish_episode()
        policy.update_parameters()

    return avg_rewards_past

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
    rewards_orig = train(0.99, 0.01, 2000, 800, "MODIFIED_JS1", False) #Gamma, lr, episodes, steps, path name
    rewards_mod = train(0.99, 0.01, 2000, 800, "ORIGINAL_JS1", True) #Gamma, lr, episodes, steps, path name
    plots(rewards_orig, rewards_mod)

#random_seed = 543
#torch.manual_seed(random_seed)
#env = gym.make('LunarLander-v2')
#env.seed(random_seed)