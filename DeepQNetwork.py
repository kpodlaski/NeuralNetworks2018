import gym
import skimage.transform
import skimage.color
import tensorflow as tf

import numpy as np

def process_observation(observation, shape):
    return skimage.transform.resize(skimage.color.rgb2gray(observation), shape)

env =gym.make("MsPacman-v0")
learning_rate= 0.001
learning_set_size =100

state = env.reset()

print(process_observation(state,[210,160]))
print(state.shape[:-1])

model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(8,8)
                                 ,input_shape=[learning_set_size,210,160]
                                 , activation='relu'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(env.action_space.n, activation='softmax'))
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=learning_rate))