import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class LQR(gym.Env):
    
    def _reset(self):
        self.state = np.array([10, 10]).reshape(2, 1)
        self.time = 0
        return self.state
    
    def __init__(self, high):
        self.high = high

        self.action_space = spaces.Box(low=-high, high=high, shape=(2,))

        self._seed()
        self.time = 0
        self.A = np.eye(2)
        self.B = np.eye(2)
        self.xi = 0.1
        self.Q1 = np.array([[self.xi, 0], [0, 1-self.xi]])
        self.Q2 = np.array([[1-self.xi, 0], [0, self.xi]])
        self.R2 = np.array([[self.xi, 0], [0, 1-self.xi]])
        self.R1 = np.array([[1-self.xi, 0], [0, self.xi]])
        
        
    def _step(self,u):
        u = u.reshape(2, 1)
        r_t = np.array([self.state.T.dot(self.Q1).dot(self.state) + u.T.dot(self.R1).dot(u), self.state.T.dot(self.Q2).dot(self.state) + u.T.dot(self.R2).dot(u)])
        next_state = self.A.dot(self.state) + self.B.dot(u)
        self.time += 1
        end = False
        if(self.time == 100):
            end = True
        self.state = next_state.reshape(2, 1)
        return next_state, -r_t.reshape(2, ), end, {}