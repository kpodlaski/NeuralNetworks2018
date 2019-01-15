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

#env =gym.make("MsPacman-v0")
env =gym.make("Pong-v0")
learning_rate= 0.001
learning_set_size =100

state = env.reset()
desired_shape_w = 170
desired_shape_h = 160

print(process_observation(state,[desired_shape_w,desired_shape_h]))


model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(8,8)
                                 ,input_shape=[desired_shape_w,desired_shape_h,1]
                                 , activation='relu'))
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(env.action_space.n, activation='softmax'))
model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])



max_games =60
epsilon = 0.9
gamma=0.8



memory = deque()

##train net
def train_model(memory, verbosity=0):
    inputs = np.zeros([len(memory),desired_shape_w,desired_shape_h,1])
    targets = np.zeros([len(memory),env.action_space.n])
    iter = 0
    #print("Memory size = " + str(len(memory)))
    #print("Prepare inputs:")
    for mem_element  in memory:
        #print(iter)
        #print(mem_element)
        state, action, state_new, reward, done = mem_element
        futureReward = np.max(model.predict(np.reshape(state_new,[1,desired_shape_w,desired_shape_h,1])))
        target = reward + gamma*futureReward
        inputs[iter] = np.reshape(state,[desired_shape_w,desired_shape_h,1])
        targets[iter] = target
        iter+=1
    #print("Train model:")
    model.fit(inputs,targets,epochs=1,verbose= verbosity)
    global epsilon
    # zmiejszane epsilon wraz z doświadczeniem
    epsilon =epsilon*.9

##Test play - fill memory
g =0
for g in range(0,max_games):
    done = False
    state = env.reset()
    #Pong starts after 18-20 moves
    prev_state = None
    for x in range(1, 20):
        prev_state = state
        state, reward, done, info = env.step(0);
    #cut height
    dropped_rows = 40
    prev_state = prev_state[dropped_rows:, :]
    prev_state = process_observation(prev_state, [desired_shape_w, desired_shape_h])
    state = state[dropped_rows:,:]
    state = process_observation(state, [desired_shape_w, desired_shape_h])
    agg_state = state + prev_state / 2
    while not done:
        #env.render()
        if random.random()<epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(model.predict(np.reshape(agg_state,[1,desired_shape_w,desired_shape_h,1])))
        state_new, reward, done, info = env.step(action)
        state_new = state_new[dropped_rows:, :]
        state_new = process_observation(state_new,[desired_shape_w,desired_shape_h])
        agg_state_new = state_new+state/2
        memory.append((agg_state, action, agg_state_new, reward, done))
        state = state_new
        agg_state = agg_state_new
        if len(memory) > 5:
            train_model(memory, verbosity=0)
            memory.clear()
        if done:
            print ("die "+str(g))
            if g % 20 == 0:
                model.save("out_files/pong_bot_"+str(g)+".qnn",overwrite=True)
                print("saved "+ str(g))
model.save("out_files/pong_bot_" + str(g) + ".qnn", overwrite=True)

