from collections import deque
import  random
#import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import gym
import skimage.transform
import skimage.color
import tensorflow as tf


import numpy as np

#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.4
#session = tf.Session(config=config)


def process_observation(observation, shape):
    return skimage.transform.resize(skimage.color.rgb2gray(observation), shape)

env =gym.make("MsPacman-v0")
learning_rate= 0.001
learning_set_size =100

state = env.reset()

print(process_observation(state,[210,160]))
print(state.shape[:-1])


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(24, kernel_size=(8,8)
                                 ,input_shape=[210,160,1]
                                 , activation='relu'))
model.add(tf.keras.layers.Conv2D(24, kernel_size=(3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(env.action_space.n, activation='softmax'))
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])



max_games =1
epsilon = 0.9
gamma=0.8


model = tf.keras.models.load_model("out_files/pack_bot_19_.qnn")


##Test play - fill memory
for g in range(0,max_games):
    done = False
    state = env.reset()
    while not done:
        env.render()
        #TODO: zmiejszane epsilon wraz z doświadczeniem
        state = process_observation(state,[210,160])
        action = np.argmax(model.predict(np.reshape(state,[1,210,160,1])))
        print(action)
        state_new, reward, done, info = env.step(action)
        state_new = process_observation(state_new,[210,160])
        state = state_new
        if done:
            print ("die "+str(g))


