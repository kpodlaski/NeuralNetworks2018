import tensorflow as tf
from tensorflow.python.keras import backend
import numpy as np


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
img_rows, img_cols = x_train.shape[1:]
print(x_train.shape)
print(x_train.shape[1:])
print(str(img_rows) + ' '+ str(img_cols))


if backend.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
            input_shape = (1, img_rows, img_cols)
else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
            input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

net = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(6,kernel_size=(5,5), strides=(1,1),
                         activation=tf.nn.relu, input_shape=input_shape),
  tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1) ),
  tf.keras.layers.Conv2D(16,kernel_size=(5,5), strides=(1,1), activation=tf.nn.relu),
  tf.keras.layers.MaxPool2D(pool_size=(2,2), strides=(1,1) ),
  tf.keras.layers.Conv2D(120,kernel_size=(5,5), strides=(1,1), activation=tf.nn.relu),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(84,activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

net.compile(optimizer='SGD',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy']#[tf.keras.metrics.mean_squared_error]
             )




net.fit(x_train, y_train, epochs=1, batch_size=100, verbose=2)
test_loss, test_acc = net.evaluate(x_test, y_test)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)
print(x_train.shape)
print(x_test[0].shape)
print(x_test.shape)
print(net.predict(x_test[0].reshape(1,28,28,1)))
print(net.predict(np.expand_dims(x_test[0],axis=0)))


