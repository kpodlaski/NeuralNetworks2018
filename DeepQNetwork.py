from collections import deque
import  random
import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
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
                                 ,input_shape=[210,160,1]
                                 , activation='relu'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(env.action_space.n, activation='softmax'))
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])



max_games =2
epsilon = 0.9
gamma=0.8



memory = deque()

##train net
def train_model(memory):
    inputs = np.zeros([len(memory),210,160,1])
    targets = np.zeros([len(memory),env.action_space.n])
    iter = 0
    print("Memory size = " + str(len(memory)))
    print("Prepare inputs:")
    for mem_element  in memory:
        #print(iter)
        #print(mem_element)
        state, action, state_new, reward, done = mem_element
        futureReward = np.max(model.predict(np.reshape(state_new,[1,210,160,1])))
        target = reward + gamma*futureReward
        inputs[iter] = np.reshape(state,[210,160,1])
        targets[iter] = target
        iter+=1
    print("Train model:")
    model.fit(inputs,targets,epochs=1)
    global epsilon
    epsilon =epsilon*.9

##Test play - fill memory
for g in range(0,max_games):
    done = False
    state = env.reset()
    while not done:
        #env.render()
        #TODO: zmiejszane epsilon wraz z doświadczeniem
        state = process_observation(state,[210,160])
        if random.random()<epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.reshape(state,[1,210,160,1])))
        state_new, reward, done, info = env.step(action)
        state_new = process_observation(state_new,[210,160])
        memory.append((state, action, state_new, reward, done))
        if len(memory) > 10:
            train_model(memory)
            memory.clear()
        if done:
            print ("die "+str(g))




