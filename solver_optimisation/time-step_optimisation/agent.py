import torch
import random
import numpy as np
from collections import deque
from streamer1d import Streamer1d
from model import CNQNet, QTrainer, Linear_QNet
from matplotlib import pyplot as plt


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

print("Using device:", device)

MAX_MEMORY = 100000
BATCH_SIZE = 10000
LR = 0.001

class Agent:
    def __init__(self, n_input, n_output):
        self.n_sims = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = CNQNet(n_output).to(device)
        #self.model = Linear_QNet(input_size=256, hidden_size=256, output_size=1).to(device)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma, device=device)

    
    def get_state(self, simulation):
        N = 128
        N_ori = len(simulation["n_e"])
        x = np.linspace(0,1,N)
        x_ori = np.linspace(0,1,N_ori)
        
        state = np.empty((3, N))

        # resample and normalize based on each channel max_val
        if simulation["efeld"].sum():
            efeld_max = np.abs(simulation["efeld"]).max()
        else:
            efeld_max = 1
        state[0,:] = np.interp(x, x_ori, simulation["n_e"]) / simulation["n_e"].max()
        state[1,:] = np.interp(x, x_ori, simulation["n_i"]) / simulation["n_i"].max()
        state[2,:] = np.interp(x, x_ori, np.abs(simulation["efeld"])) / efeld_max

        return state
    

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        #states, actions, rewards, next_states, dones = zip(*mini_sample)
        #self.trainer.train_step(states, actions, rewards, next_states, dones)
        for state, action, reward, next_state, done in mini_sample:
            self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # List of time steps to choose from
        dts = [9.55555556e-13, 2.74444444e-12, 1.46666667e-12, 1.21111111e-12,
            2.23333333e-12, 7.00000000e-13, 1.72222222e-12, 3.00000000e-12,
            2.48888889e-12, 1.97777778e-12]
        # random moves: tradeoff exploration / exploitation
        end_eps = 0.01
        n_explor = 100
        self.epsilon = 10#n_explor - self.n_sims
        if self.epsilon <= (n_explor * end_eps):
            self.epsilon = (n_explor * end_eps)
        n = len(dts)
        if random.randint(0, n_explor) < self.epsilon:
            #idx = random.randint(0, n-1)
            a = np.random.rand(1,n)
            move = np.argmax(a)
            pred_dt = dts[move]
            action = a[0]
            #print('action Rnd =', action)
        else:
            state0 = torch.tensor(state, dtype=torch.float).to(device)
            #print('state0 shape =', state0.shape)
            #print(self.model(state0))
            a = self.model(state0)
            move = torch.argmax(self.model(state0))
            #print(state0.shape)
            #print(a)
            pred_dt = dts[move]
            action = a.cpu().detach().numpy()[0]
            #print('action RL =', action)
            #print('pred_dt =', pred_dt)

        return pred_dt, action