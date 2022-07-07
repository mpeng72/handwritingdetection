import tensorflow
import numpy as np
import keras
from keras import models
from keras import layers
from extra_keras_datasets import emnist


#define training constants
num_classes = 62
epochs = 10
batch_size = 128
input_shape = (28,28,1)

model = models.Sequential()
model.add(layers.Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(num_classes, activation='softmax'))
model.summary()

(train_images, train_labels), (test_images, test_labels) = emnist.load_data(type = 'byclass')

#Declare the size of the training and test sets (w&h of 28 + depth of 1)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

#convert all values to floats and change image to b&w
train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(learning_rate = 0.001),metrics=['accuracy'])
model.fit(train_images, train_labels, epochs = epochs, batch_size=batch_size)

model.save('models/emnist.h5')

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
