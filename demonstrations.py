"""
Demonstration data structure to be saved.
Must include:
    - Name of the environment
    - Version of the environment
    - Timestamp array
    - Timestep array
    - State/Observation
    - Action
    - Reward

Returns a dictionary of required demonstration fields
"""
import pickle
from simplePG import SimplePG
from lunar_lander import LunarLander
import pygame

class Demonstration():
    
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

    
    