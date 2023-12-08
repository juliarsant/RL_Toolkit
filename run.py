"""
Julia Santaniello
Tufts University, MuLIP
HITL Reinforcement Learning with Brain-Computer Interfaces

Main module that streamlines all games, agents, algorithms and data collection, 
plotting and csv saving.
"""


from Algorithms import SimplePG
from Data import Demonstration, DataClass
from Games import LunarLander, PixelCopter, FlappyBird
import Agents
import BCI

NO_HITL = True
USE_BCI = False

LL = LunarLander()
PC = PixelCopter()

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
    pass

def main_bcihitl():
    pass

if __name__ == "__main__":
    run()