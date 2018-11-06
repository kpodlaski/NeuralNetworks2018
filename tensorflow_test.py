import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
mnist = tf.keras.datasets.mnist

(train_inputs, train_labels), (test_inputs, test_labels) = mnist.load_data()
train_inputs, test_inputs = train_inputs / 255.0, test_inputs / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28,28)),
  tf.keras.layers.Dense(30,activation=tf.nn.relu),#512,tf.nn.sigmoid
  #tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.sigmoid)# tf.nn.softmax, tf.nn.relu)
])
model.compile(optimizer=tf.train.GradientDescentOptimizer(0.5),
              loss='sparse_categorical_crossentropy',#tf.keras.losses.mean_absolute_error , #
              metrics=['accuracy']#[tf.keras.metrics.mean_squared_error]
             )

model.fit(train_inputs, train_labels, epochs=10, batch_size=100)
test_loss, test_acc = model.evaluate(test_inputs, test_labels)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

predictions = model.predict(test_inputs)
print("Result : ")
good_response = 0
for elem in range(0,len(test_inputs)):
    if np.argmax(predictions[elem]) == test_labels[elem]:
        good_response+=1
print(predictions[len(test_inputs)-1])
#import matplotlib.pyplot as plt
plt.figure(figsize=(28,28))
plt.xticks([])
plt.yticks([])
plt.imshow(test_inputs[len(test_inputs)-1], cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()
print(np.argmax(predictions[len(test_inputs)-1]))
print(test_labels[len(test_inputs)-1])
print(good_response/len(test_inputs)*100.0)

