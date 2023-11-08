"""
Julia Santaniello
06/25/23

For training a single policy. Input hyperparamters and saved policy name.
"""

#import pygame
import pygame
from simplePG import SimplePG
import torch.optim as optim
from lunar_lander import LunarLander 
import matplotlib.pyplot as plt
from demonstrations import Demonstration
import time
import pickle
import random

GAMMA = 0.99
NUM_DEMOS = 2
EPISODES = 5000
LR = 0.01 #learning rate
STEPS = 500 #steps in each episode
ENVIRONMENT_NAME = "LunarLander"
ENVIRONMENT_VERSION = "Modified"

record_demos = True


def train_with_demonstrations(gamma, lr, eps, step):
    file = open('./data/demos/demos_10.pickle', 'rb')
    demo_dict = pickle.load(file)
    file.close()

    env = LunarLander()
    policy = SimplePG(num_actions = 4, input_size = 11, hidden_layer_size=11, 
                      learning_rate=0.1, gamma=0.99, decay_rate=0.9, greedy_e_epsilon=0.1, 
                      random_seed=10) #11 state elements in modified LunarLander
    
    policy.set_explore_epsilon(0.1)
    demo_eps = len(demo_dict)
    avg_rewards_past = [] 

    for i in range(demo_eps + eps):
        if i >= 100 and i < 200:
            steps = demo_dict[i-100]["steps"]
            seed = demo_dict[i-100]["seed"]
            state, _ = env.reset(seed=seed)
        else:
            steps = step
            state, _ = env.reset()

        running_reward = 0

        for j in range(steps):
            #pick an action
            if i >= 100 and i<200:
                human_action = demo_dict[i-100]["actions"][j]
                policy.process_step(demo_dict[i-100]["states"][j], True)
                policy.save_human_action(human_action)
                next_state, reward, done, _, win = env.step(human_action)
            else:
                action = policy.process_step(state, False)
                next_state, reward, done, _, win = env.step(action)

            policy.save_rewards(reward)
            running_reward += reward
            state = next_state

            if done:
                break
    
        if i % 20 == 0:
            average_reward = running_reward/20
            avg_rewards_past.append(average_reward)
            print('Episode {}\tlength: {}\treward: {}'.format(i, j, average_reward))
            running_reward = 0
        
        if i >= 100 and i < 200:
            policy.finish_episode(True)
        else:
            policy.finish_episode(False)

        policy.update_parameters()

    return avg_rewards_past
        
    



"""
Train_without_demonstrations():
Purpose: Train policy without the use of human demonstrations
Returns: List of average rewards for each episode
"""
def train_without_demonstrations(gamma, lr, eps, steps, num_demos:int):
    
    # if num_demos < 0:
    #     env = LunarLander(render_mode="human") #modified LunarLander
    #     human = True
    # else:
    env = LunarLander() #modified LunarLander
    human = False
    policy = SimplePG(num_actions = 4, input_size = 11, hidden_layer_size=11, 
                      learning_rate=lr, gamma=gamma, decay_rate=0.9, greedy_e_epsilon=0.1, 
                      random_seed=10) #11 state elements in modified LunarLander
    
    policy.set_explore_epsilon(0.1)
    
    rewards_past = []
    avg_rewards_past = []
    running_reward = 0

    for i_episode in range(0, eps + num_demos):
        episode_rewards = 0
        state, _ = env.reset(seed=10)
        # #keeping track of demonstrations
        # if num_demos <= 0:
        #     human = False
        #     env = LunarLander()
        #     state, _ = env.reset()
        # if human:
        #     num_demos -= 1

        for t in range(steps):
            #pick an action
            if not human:
                action = policy.process_step(state, False)
                state, reward, done, _, win = env.step(action)
            else:
                human_action = human_play()
                policy.process_step(state, True)
                policy.save_human_action(human_action)
                state, reward, done, _, win = env.step(human_action)

            policy.save_rewards(reward)
            running_reward += reward
            episode_rewards += reward

            if done:
                break
        
        rewards_past.append(episode_rewards)

        if i_episode % 20 == 0:
            average_reward = running_reward/20
            avg_rewards_past.append(average_reward)
            print('Episode {}\tlength: {}\treward: {}'.format(i_episode, t, average_reward))
            running_reward = 0
    
        policy.finish_episode(human)
        policy.update_parameters()

    return avg_rewards_past


"""
Plots()
Purpose: Plot policy trained with or without demonstrations
Return: None; Print plots
"""
def plots(rewards_demos, rewards_no_demos):
    num_demos = len(rewards_demos) - len(rewards_no_demos)

    #X-axis
    iterations = range(0,len(rewards_no_demos)*20,20)

    #Plot
    plt.plot(iterations, rewards_demos[num_demos:])
    plt.plot(iterations, rewards_no_demos)
    plt.ylabel("Average Return")
    plt.xlabel("Iterations")
    plt.legend(["With Human", "Without Human"])
    plt.show()


"""
DemostrationsOnly()

Purpose: Collects data from human demonstrations, saves them in a dictionary, 
         and pickles them. Saved in .pickle files in the "./data/demos/*" folder
Input: Gamma rate, learning rate, trajectory step constraint, number of 
       demonstrations desired.
Returns: A python dictionary of demonstration data. Keys of 0:len(demo)-1 are saved. 
         Each key holds all the information required to replay the demo for visual 
         examples or training purposes.
"""
def demonstrations_only(gamma, lr, steps, num_demos:int):
    env = LunarLander(render_mode="human") #modified LunarLander game
    human = True #Demonstrations are occuring, render the game

    demonstrations_dict = {} #dictionary of demonstrations
    
    #Create policy
    policy = SimplePG(num_actions = 4, input_size = 11, hidden_layer_size=11, 
                      learning_rate=lr, gamma=gamma, decay_rate=0.9, greedy_e_epsilon=0.1, 
                      random_seed=10) #11 state elements in modified LunarLander
    
    #set epsilon value
    policy.set_explore_epsilon(0.1)

    #Rewards per epsiode saved
    rewards_per_episode = [] 

    #for each demonstration desired
    for i_episode in range(0, num_demos):
        state, _ = env.reset(seed=i_episode) #reset
        timestamps, action_list = [], [] #timestamps
        running_reward = 0

        for t in range(steps):
            #pick an action
            human_action = human_play()
            action_list.append(human_action)

            #timestamp update
            current_time = time.time()
            timestamps.append(current_time)

            #Process human action
            policy.process_step(state, True)

            #Save action
            policy.save_human_action(human_action)

            #Return state, reward
            state, reward, done, _, _ = env.step(human_action)

            policy.save_rewards(reward)

            running_reward += reward

            if done:
                break

        final_episode_actions = action_list
        final_episode_rewards = policy._drs
        final_episode_states = policy._xs
        final_episode_steps = t


        assert(len(timestamps) == len(final_episode_actions) == len(final_episode_rewards) == len(final_episode_states))
        
        policy.finish_episode(human)
        policy.update_parameters() 
        rewards_per_episode.append(running_reward)
        DemoClass = Demonstration(ENVIRONMENT_NAME, ENVIRONMENT_VERSION, i_episode, final_episode_steps, timestamps, final_episode_states, final_episode_actions, final_episode_rewards)
        finished_demo = DemoClass.save_demonstration()
        demonstrations_dict[i_episode] = finished_demo

    return demonstrations_dict

def play_demonstrations():
    file = open('./data/demos/demos_10.pickle', 'rb')
    demo_dict = pickle.load(file)
    file.close()

    env = LunarLander(render_mode="human")
    policy = SimplePG(num_actions = 4, input_size = 11, hidden_layer_size=11, 
                      learning_rate=0.1, gamma=0.99, decay_rate=0.9, greedy_e_epsilon=0.1, 
                      random_seed=10) #11 state elements in modified LunarLander

    for i in range(len(demo_dict)):
        steps = demo_dict[i]["steps"]
        seed = demo_dict[i]["seed"]
        state, _ = env.reset(seed=seed)

        for j in range(steps):
            action = demo_dict[i]["actions"][j]

            #Return state, reward
            state, reward, done, _, _ = env.step(action)

            policy.save_rewards(reward)

            if done:
                break

"""
Human_play()
Purpose: allows human to choose actions based on the state
Returns: Chosen Action (int)
"""
def human_play():
    pressed_keys = pygame.key.get_pressed()

    if pressed_keys[pygame.K_LEFT]: #left
        return 1
    elif pressed_keys[pygame.K_UP]: #up
        return 2
    elif pressed_keys[pygame.K_RIGHT]: #right
        return 3
    
    return 0 #do nothing

    

if __name__ == "__main__":
    rewards_with_demos = train_with_demonstrations(GAMMA, LR, EPISODES, STEPS) 
    rewards_no_demos = train_without_demonstrations(GAMMA, LR, EPISODES, STEPS, 100) #Gamma, lr, episodes, steps, path name, lunarlander version
    plots(rewards_no_demos, rewards_with_demos)



#random_seed = 543
#torch.manual_seed(random_seed)
#env = gym.make('LunarLander-v2')
#env.seed(random_seed)

"""To save a model"""
        # saving the model if episodes > 999 OR avg reward > 200 
        #if i_episode > eps-1:
        #    torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(title))
        
        # if average_reward > 210:
        #     torch.save(policy.state_dict(), './preTrained/LunarLander_{}.pth'.format(title))
        #     print("########## Solved! ##########")
        #     #test(name='LunarLander_{}}.pth')#.format(title))
        #     break

#name of environment, number of steps, 
#timestamp, reward, action, state/observation, name of game