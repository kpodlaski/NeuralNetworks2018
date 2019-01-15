# Source
# https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
import random

import gym
import numpy as np

env = gym.make('FrozenLake-v0')

#Initialize Q table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])
# Set learning parameters
lr = .81
y = .95
num_episodes = 2000
#create lists to contain total rewards and steps per episode
#jList = []
rList = []
random_decision_probability = 0.3
random.seed()
for i in range(num_episodes):
    #Reset environment and get first new observation
    state = env.reset()
    totalReward = 0
    done = False
    step = 0
    #The Q-Table learning algorithm
    while step < 99:
        step+=1
        #Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        if random.random()>random_decision_probability:
            action = np.argmax(Q[state, :])
        else:
            action = env.action_space.sample()
        #Get new state and reward from environment
        new_state, reward, done, info = env.step(action)
        #Update Q-Table with new knowledge
        Q[state, action] = Q[state, action] + lr * (reward + y * np.max(Q[new_state, :]) - Q[state, action])
        totalReward += reward
        state = new_state
        if done == True:
            break
    #jList.append(j)
    rList.append(totalReward)
print ("Score over time: " +  str(sum(rList)/num_episodes))
print ("Final Q-Table Values")
print (Q)

test_episodes = 6
print ("Testing Q Table")
for i in range(0, test_episodes):
    state = env.reset()
    step = 0
    done = False
    # The Q-Table learning algorithm
    while step < 99:
        step += 1
        action = np.argmax(Q[state, :] )
        state, reward, done, info = env.step(action)
        if done == True:
            print ("Final state:"+str(state)+ " reward:"+str(reward)+" :in steps"+str(step)+": "+str(info))
            break
print("Next Episode ")