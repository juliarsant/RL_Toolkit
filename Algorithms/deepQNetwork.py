import keras
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.optimizers import Adam
from Games.ple.games.pixelcopter import Pixelcopter
from Games.ple import PLE
import pygame
import numpy as np
import time
from IPython.display import clear_output
import random
import matplotlib.pyplot as plt
import datetime

class DeepQNetwork():
    def __init__(self, p, game, num_actions, obs_space, epochs, gamma, epsilon, buffer=400, replay = [], batchSize = 200, rewards=None):
        self.epochs = epochs
        self.rewards = np.zeros((1,epochs))[0]
        self.gamma = gamma
        self.epsilon = epsilon
        self.batchSize = batchSize
        self.buffer = buffer
        self.replay = replay
        self.p = p
        self.game = game
        self.num_actions = num_actions
        self.obs_space = obs_space

        self.model_init()

    def model_init(self):

        model = Sequential(
            Dense(64, input_dim=self.obs_space, activation="relu"),
            Dense(64, activation="relu"),
            Dense(64, activation="relu"),
            Dense(16, activation="linear"))
        
        adam = Adam(lr=0.0001)

        model.compile(loss='mse', optimizer=adam)
        
        self.model = model

    def train(self):
        
        for i in range(self.epochs):

            self.p.reset_game()

            while (not self.p.game_over()):
                state = self.game.getGameState()
                stateLst = np.array([[state[k] for k in state]])
                qval = self.model.predict(stateLst)
                if (random.random() < self.epsilon):
                    action = np.random.randint(0,2)
                    print("here ", action)
                else:
                    action = np.argmax(qval)

                actionList = self.p.getActionSet()
                reward = self.p.act(actionList[action])

                newState = self.game.getGameState()
                newStateLst = np.array([[newState[k] for k in state]])

                if (len(self.replay) < self.buffer):
                    self.replay.append((stateLst, action, reward, newStateLst))
                else:
                    if h < self.buffer-1:
                        h += 1
                    else:
                        h = 0
                    self.replay[h] = (stateLst, action, reward, newStateLst)
                    minibatch = random.sample(self.replay, self.batchSize)
                    X_train = np.empty((0,5))
                    y_train = np.empty((0,2))

                    for memory in minibatch:
                        old_state, action, reward, new_state = memory
                        oldQ = self.model.predict(old_state)
                        newQ = self.model.predict(newStateLst)
                        maxQ = np.max(newQ)

                        if self.p.game_over():
                            update = reward
                        else:
                            update = reward + (self.gamma * maxQ)
                        y = np.copy(oldQ)
                        y[0][action] = update
                        X_train = np.append(X_train, old_state, axis=0)
                        y_train = np.append(y_train, y, axis=0)

                    self.model.fit(X_train, y_train, batch_size=self.batchSize)

                pygame.display.update()
                print(i)
                clear_output(wait=True)


            self.rewards[i] = self.p.score()
            i += 1
            if self.epsilon > 0.0001: #decrement epsilon over time
                self.epsilon -= (1/self.epochs)

            if i % 5000 == 0:    
                f = "w_" + str(datetime.datetime.now()).replace(" ", "_").replace(":", "_") + ".h5py"
                self.model.save(f)

    def evaluate(self):
        self.p.init()
        self.game.screen = pygame.display.set_mode(self.game.getScreenDims(), 0, 32)
        self.game.clock = pygame.time.Clock()

        for i in range(self.epochs):
        
            self.p.reset_game()

            while (not self.p.game_over()):
    
                state = self.game.getGameState()
                state_array = np.array([[state[z] for z in state]])
                aprobs = self.model.predict(state_array)
                action = np.argmax(aprobs)
                action_array = self.p.getActionSet()
                reward = self.p.act(action_array[action])

                pygame.display.update()


    evaluate(1000)