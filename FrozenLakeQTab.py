#Source
#https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
import random

import gym
import numpy as np

env = gym.make("FrozenLake-v0")

Q =np.zeros([env.observation_space.n, env.action_space.n])
number_of_episodes = 2000
max_steps =100
random_decision_probability = 0.240
gamma=0.95
lr = .8

for i  in range (0, number_of_episodes):
    done = False
    state = env.reset()
    totalReward = 0
    step = 0
    while step<max_steps:
        step+=1
        if random.random()>random_decision_probability:
            action = np.argmax(Q[state, :])
        else:
            action = env.action_space.sample()
        #action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        new_state, reward, done, info = env.step(action)
        Q[state,action]= Q[state,action] + lr*(reward
                    + gamma*np.max(Q[new_state,:]) - Q[state,action])
        totalReward +=reward
        state=new_state
        if done==True: break
        #env.render()


test_episodes = 10
sum_of_rewards = 0
for i  in range (0, test_episodes):
    done = False
    state = env.reset()
    totalReward = 0
    step = 0
    while not done and step<max_steps:
        step+=1
        action = np.argmax(Q[state, :])
        new_state, reward, done, info = env.step(action)
        totalReward +=reward
        state=new_state
        #env.render()
    sum_of_rewards +=totalReward
    print("Last state: " + str(state))
print("Tot: "+str(sum_of_rewards/test_episodes))

