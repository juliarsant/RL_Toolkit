"""
Julia Santaniello
06/25/23

HITL Training
"""

import pygame
import matplotlib.pyplot as plt
import time
import pickle
from Data.demonstrations import Demonstration

GAMMA = 0.99
NUM_DEMOS = 50
EPISODES = 5000
LR = 0.01 #learning rate
STEPS = 500 #steps in each episode

record_demos = True

class HumanInTheLoopTrain():
    def __init__(self, environment, env_name, algorithm, episodes, steps, gamma, alpha, seed, num_demos, name):
        self.env_name = env_name
        self.steps = steps
        self.env = environment
        self.policy = algorithm
        self.eps = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.num_demos = num_demos
        self.demo_name = name
        self.seed = seed
        
    """
    Human_play()
    Purpose: allows human to choose actions based on the state
    Returns: Chosen Action (int)
    """
    def human_play(self):
        pressed_keys = pygame.key.get_pressed()

        if pressed_keys[pygame.K_LEFT]: #left
            return 1
        elif pressed_keys[pygame.K_UP]: #up
            return 2
        elif pressed_keys[pygame.K_RIGHT]: #right
            return 3
        
        return 0 #do nothing
    def start():
        input("Press Enter to Start Demonstrations: ")
        print("Starting in 3...")
        time.time(1)
        print("2...")
        time.time(1)
        print("1...")
        time.time(1)
        print("Start!")
        print("")

    def run(self):
        self.start()
        self.demonstrations_only(self.demo_name)
        print("Thank you!")
        time.time(15)
        print("Starting agent training...")
        self.train_with_demonstrations()
    
    def train_with_demonstrations(self):
        file = open('./data/demos/{}}.pickle'.format(self.demo_name), 'rb')
        demo_dict = pickle.load(file)
        file.close()

        env = self.env
        policy = self.policy

        demo_eps = len(demo_dict)

        avg_rewards_past = [] 

        for i in range(demo_eps + self.eps):
            if i >= 100 and i < 200:
                steps = demo_dict[i-100]["steps"]
                seed = demo_dict[i-100]["seed"]
                state, _ = env.reset(seed=seed)
            else:
                steps = self.step
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
    def train_without_demonstrations(self, num_demos):
        env = self.env 
        policy = self.policy
        
        rewards_past = []
        avg_rewards_past = []
        running_reward = 0

        for i_episode in range(0, self.eps + num_demos):
            episode_rewards = 0
            state, _ = env.reset(seed=10)

            for t in range(steps):
                #pick an action
                if not human:
                    action = policy.process_step(state, False)
                    state, reward, done, _, win = env.step(action)
                else:
                    human_action = self.human_play()
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
    def demonstrations_only(self, num_demos):
        env = self.env #modified LunarLander game
        human = True #Demonstrations are occuring, render the game

        demonstrations_dict = {"demo_name": self.demo_name} #dictionary of demonstrations
        
        #Create policy
        policy = self.policy
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
            DemoClass = Demonstration(environment_name=self.env_name, environment_version=None, seed=self.seed, steps= final_episode_steps, timestamps= timestamps, states=final_episode_states, actions=final_episode_actions, rewards=final_episode_rewards)
            finished_demo = DemoClass.save_demonstration()
            demonstrations_dict[i_episode] = finished_demo

        return demonstrations_dict

    def play_demonstrations(self):
        file = open('./data/demos/demos_10.pickle', 'rb')
        demo_dict = pickle.load(file)
        file.close()

        env = self.env
        policy = self.policy

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

        

if __name__ == "__main__":
    pass