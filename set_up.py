from Algorithms import SimplePG
from Games import LunarLander, PixelCopter, FlappyBird
from Agents.train import Train
from Agents.bci_train import BCITrain
from Agents.hitl_train import HumanInTheLoopTrain

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
        self.env = self.choose_environment()
        self.action_space, self.obs_space, self.obs_space_type = self.extract_paramters()
        self.algorithm = self.choose_algorithm()

        if self.params["bci"] == True:
            pass
        elif self.params["demonstrations_only"]["boolean"] == True:
            agent = HumanInTheLoopTrain(self.env, self.env_name, self.algorithm, self.episodes, self.steps, self.gamma, self.alpha, self.seed, self.params["demonstrations_only"]["num_demos"], self.params["demonstrations_only"]["name"])

        

    def choose_environment(self):
        if self.env_name == "pixelcopter":
            env = PixelCopter()
        elif self.env_name == "lunar_lander":
            env = LunarLander()
        elif self.env_name == "flappy_bird":
            env = FlappyBird()

        return env

    def extract_paramters(self):
        return self.env.extract_parameters()

    def choose_algorithm(self):
        if self.algorithm_name == "ppo":
            assert self.obs_space_type == "int"
            algo = SimplePG(self.action_space, self.obs_space, self.obs_space, self.params[0], self.params[1], self.params[2], self.params[3],self.params[4])
        return algo


        