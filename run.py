"""
Julia Santaniello
Tufts University, MuLIP
HITL Reinforcement Learning with Brain-Computer Interfaces

Main module that streamlines all games, agents, algorithms and data collection, 
plotting and csv saving.
"""


from Data import Demonstration, DataClass
from set_up import SetUp


NO_HITL = True
USE_BCI = False

"""
REQUIRED INPUTS HERE:
"""
params = {"gamma": 0.99,
          "learning_rate": 0.1,
          "decay_rate": 0.001,
          "epsilon": 0.1,
          "seed": 10,
          "bci": False,
          "demonstrations_only": {"boolean": True, "num_demos": 10, "name": "PARTICIPANT_001_12122023546"},
          "episodes": 1000,
          "steps": 500,
          "algorithm": "ppo",
          "environment": "lunar lander"
          }

def run():
    if NO_HITL == True:
        main()
    elif USE_BCI == True:
        main_bcihitl()
    else:
        main_hitl()
    pass

def main():
    pass

def main_hitl():
    setup = SetUp(params)

    pass

def main_bcihitl():
    pass

if __name__ == "__main__":
    run()