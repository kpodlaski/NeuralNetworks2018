from matches_encoding import mapping
import tensorflow as tf
import numpy as np


net = None

#SIMPLE NET 28% efectivity
net = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(5,4)),
  tf.keras.layers.Dense(10,activation=tf.nn.sigmoid),#tf.nn.relu
  tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)# tf.nn.softmax)
])

# net = tf.keras.models.Sequential([
#  tf.keras.layers.Flatten(input_shape=(5,4)),
#  tf.keras.layers.Dense(10,activation=tf.nn.relu),
#  tf.keras.layers.Dense(10, activation=tf.nn.softmax)# )
# ])


net.compile(optimizer=tf.train.GradientDescentOptimizer(0.2),
              loss=tf.keras.losses.binary_crossentropy , #'sparse_categorical_crossentropy',#
              metrics=['accuracy'])

from matches_encoding import mapping

training_set = list()
training_labels = list()
for i in range (0,100):
    _exp_output = i%10
    _sig = mapping[_exp_output]
    training_set.append(_sig)
    lab = np.zeros(10)
    lab[_exp_output]=1
    training_labels.append(lab)

#Train_network
net.fit(np.array(training_set), np.array(training_labels), epochs=20, batch_size=10)

#Test_network
test_set = list()
test_labels = list()
for i in range (0,1000):
    _eo = np.zeros(10)
    _exp_output = i%10
    test_set.append(mapping[_exp_output])
    lab = np.zeros(10)
    lab[_exp_output] = 1
    test_labels.append(lab)
loss, acc = net.evaluate(np.array(test_set),np.array(test_labels))
print("Loss :"+ str(loss))
print("Acc :"+ str(acc))
results = net.predict(np.array(test_set[0:10]))
print("---------")
for i in range(0,10):
    print(np.array_str(np.array(test_set[i])))
    print("Result "+str(np.argmax(results[i])))
    print("Exp "+str(i))
    print("---------")
