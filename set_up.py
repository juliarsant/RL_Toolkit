from Algorithms.dqn import DQN
from Algorithms.simplePG import SimplePG
from Games import LunarLander, PixelCopter, FlappyBird
from Agents.train import Train
from Agents import train2
from Agents.bci_train import BCITrain
from Agents.hitl_train import HumanInTheLoopTrain
from save_demos import SaveDemos
from Games.ple import PLE

class SetUp():
    def __init__(self, params) -> None:
        self.env_name = params["environment"]
        self.algorithm_name = params["algorithm"]
        self.gamma = params["gamma"]
        self.epsilon = params["epsilon"]
        self.lr = params["learning_rate"]
        self.episodes = params["episodes"]
        self.steps = params["steps"]
        self.alpha = params["alpha"]
        self.seed = params["seed"]

        self.params = params


    def set_up(self):
        self.p, self.game = self.choose_environment()
        self.action_space, self.obs_space, self.obs_space_type = self.extract_paramters()
        self.algorithm = self.choose_algorithm()

        if self.params["bci"] == True:
            print("BCI")
            pass
        elif self.params["demonstrations_only"]["train"] == True:
            print("HITL")
            agent = HumanInTheLoopTrain(self.env, self.env_name, self.algorithm, self.algorithm_name, self.episodes, self.steps, self.gamma, self.alpha, self.seed, self.params["demonstrations_only"]["num_demos"], self.params["demonstrations_only"]["name"])
            agent.run()
        elif self.params["demonstrations_only"]["demos_only"] == True:
            agent = SaveDemos(self.env, self.env_name, self.algorithm, self.algorithm_name, self.episodes, self.steps, self.gamma, self.alpha, self.seed, self.params["demonstrations_only"]["num_demos"], self.params["demonstrations_only"]["name"])
        else:
            print("train")
            #agent = Train(self.env, self.env_name, self.algorithm, self.episodes, self.steps, self.gamma, self.alpha, self.seed, self.params["environment"])
            train2.train(self.p, self.game, self.action_space, self.obs_space, self.episodes, self.gamma, self.epsilon)
            #agent.run()

        

    def choose_environment(self):
        if self.env_name == "pixelcopter":
            game = PixelCopter(width=256, height=256)
            p = PLE(game, fps=150, display_screen=self.params["demonstrations_only"]["demos_only"])
        elif self.env_name == "lunar lander":
            if self.params["demonstrations_only"]["demos_only"]:
                env = LunarLander(render_mode="human")
            else: env = LunarLander()
        elif self.env_name == "flappy_bird":
            env = FlappyBird()

        return game, p

    def extract_paramters(self):
        return self.env.extract_parameters()

    def choose_algorithm(self):
        if self.algorithm_name == "ppo":
            assert self.obs_space_type == "int"
            algo = SimplePG(self.action_space, self.obs_space, self.obs_space, self.params[0], self.params[1], self.params[2], self.params[3],self.params[4])
        elif self.algorithm_name == "dqn":
            assert self.obs_space_type == "int"
            algo = DQN(self.env, self.action_space, self.obs_space, self.obs_space, self.params[0], self.params[1], self.params[2], self.params[3],self.params[4])
        elif self.algorithm_name == "dqn_cnn":
            algo = None
        return algo


        