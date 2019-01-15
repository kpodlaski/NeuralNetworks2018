from collections import deque
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
import skimage.transform
import skimage.color
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import time
import numpy as np

def process_observation(observation, shape):
    return skimage.transform.resize(skimage.color.rgb2gray(observation), shape)

env =gym.make("Pong-v0")

env.reset()
size = env.action_space
old_state = []
state = []
for x in range(1,50):
    old_state = state
    state, reward, done, info = env.step(0);
    if done: env.reset()
    env.render()
    print(x)


print (size)

env.close()
#plt.plot(state)
#plt.show()
print(state.shape)

frame_shape = [210, 160]
obs = process_observation(state, frame_shape)
old_obs = process_observation(old_state, frame_shape)

big_obs = np.concatenate((old_obs, obs), axis=1)
print(big_obs.shape)

big_obs_rg = np.zeros((2,210,160))
big_obs_rg[0] = obs
big_obs_rg[1] = old_obs

big_obs_2 = obs +old_obs/2

print("shape" + str(np.shape(big_obs_2)))

big_obs_2 = big_obs_2[40:, :]
print(len(big_obs))
print(len(big_obs[0]))
#env.render()
fig = plt.figure(figsize=(170,160))
plt.xticks([])
plt.yticks([])
plt.imshow(big_obs_2,cmap='gray')
plt.colorbar()
plt.grid(False)
fig.savefig('fig.png')
print("show image")
#plt.show()
