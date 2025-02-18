# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 20:38:02 2022

@author: Adithya Raj Mishra
"""

from tensorflow import keras
import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import os

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
batch_size = 32
n_episodes = 1000
output_dir = 'model_output/cartpole/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
    
    def _build_model(self):
        model = Sequential()
        model.add(Dense(32, activation ='relu', input_dim=self.state_size))
        model.add(Dense(32,activation='relu'))
        model.add(Dense(self.action_size,activation='linear'))
        model.compile(loss= 'mse', optimizer=Adam(lr=self.learning_rate))
        return model   
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state,action,reward,next_state,done))
        
    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward+ self.gamma*np.amax(self.model.predict(next_state)[0]) )
            target_f = self.model.predict(state) 
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            if(self.epsilon > self.epsilon_min):
                self.epsilon *=self.epsilon_decay
                
    def act(self, state):
        if(np.random.rand() <=self.epsilon):
            return random.randrange(self.action_size)
        act_values=self.model.predict(state)
        return np.argmax(act_values[0])
    
    def save(self,name):
        self.model.save_weights(name)
    
    def load(self, name):
        self.model.load_weights(name)
        
    def saveModel(self,name):
        self.model.save(name)
        
        
        
agent= DQNAgent(state_size, action_size)
agent.load(output_dir +"weights_ 350.hdf5")
agent.saveModel(output_dir)
'''for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    done = False
    time = 0
    while not done:
        #env.render()
        action = agent.act(state)
        next_state, reward, done, _ =env.step(action)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done :
            print("episode : {}/{} , score: {}, e:{:.2}".format(e, n_episodes-1, time, agent.epsilon))
        time+=1
    if (len(agent.memory) > batch_size):
        agent.train(batch_size)
    if((e%50) ==0):
        agent.save(output_dir + "weights_" + '{0:4d}'.format(e)+".hdf5")'''
        
    

'''state=env.reset()
done = False
time = 0
while not done:
    env.render()
    action = agent.act(state)
    next_state, reward, done, _ =env.step(action)
    
    
    state = next_state
    if done :
        print("episode : {}/{} , score: {}, e:{:.2}".format(0, n_episodes-1, time, agent.epsilon))
    time+=1
    
    '''