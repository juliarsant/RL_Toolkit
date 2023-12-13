from Data.demonstrations import Demonstration
import time
import pygame

class SaveDemos():
    def __init__(self, environment, env_name, algorithm_name, algorithm, episodes, steps, gamma, alpha, seed, num_demos, name):
        self.env_name = env_name
        self.steps = steps
        self.env = environment
        self.policy = algorithm
        self.algorithm_name = algorithm_name
        self.eps = episodes
        self.gamma = gamma
        self.alpha = alpha
        self.num_demos = num_demos
        self.demo_name = name
        self.seed = seed

        self.env.init()
    
    def run(self):
        dict = self.demonstrations_only()
    
    def human_play(self):
        pressed_keys = pygame.key.get_pressed()
        if self.env_name == "lunar lander":
            if pressed_keys[pygame.K_LEFT]: #left
                return 1
            elif pressed_keys[pygame.K_UP]: #up
                return 2
            elif pressed_keys[pygame.K_RIGHT]: #right
                return 3
            return 0 #do nothing
        elif self.env_name == "pixelcopter":
            if pressed_keys[pygame.K_w]: #left
                return 1
            return 0

    def demonstrations_only(self):
        demonstrations_dict = {"demo_name": self.demo_name, "algorithm": self.algorithm_name} #dictionary of demonstrations
        
        self.policy.set_explore_epsilon(0.1)

        #Rewards per epsiode saved
        rewards_per_episode, steps_per_episode = [], []

        #for each demonstration desired
        for i_episode in range(0, self.num_demos):
            state = self.env.reset_game(seed=i_episode) #reset
            timestamps, action_list = [], [] #timestamps
            running_reward = 0

            for t in range(self.steps):
                #pick an action
                human_action = self.human_play()
                action_list.append(human_action)

                #timestamp update
                current_time = time.time()
                timestamps.append(current_time)

                #Process human action
                _ = self.policy.process_step(state, False)

                #Save action
                self.policy.save_human_action(human_action)

                #Return state, reward
                state, reward, done, _ = self.env.step(human_action)

                self.policy.save_rewards(reward)

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
    
    class DemoClass():
        def __init__(self, environment_name, environment_version, seed, 
                    steps, timestamps, states, actions, rewards):

            self.environment_name = environment_name
            self.environment_version = environment_version
            self.seed = seed
            self.steps = steps
            self.timestamps = timestamps
            self.states = states
            self.actions = actions
            self.rewards = rewards

            self.state_dict = {}

            assert(len(rewards) == len(states) == len(timestamps) == len(actions))

        def save_demonstration(self):
            self.state_dict["environment_name"] = self.environment_name
            self.state_dict["environment_version"] = self.environment_version
            self.state_dict["seed"] = self.seed
            self.state_dict["steps"] = self.steps
            self.state_dict["timestamps"] = self.timestamps
            self.state_dict["states"] = self.states
            self.state_dict["actions"] = self.actions
            self.state_dict["rewards"] = self.rewards

            return self.state_dict