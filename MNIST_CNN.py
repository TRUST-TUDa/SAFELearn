import numpy as np
import keras 
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, Flatten, BatchNormalization
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()

img_rows, img_cols = 28, 28
x_train = x_train.reshape(-1, img_rows, img_cols, 1)
x_test = x_test.reshape(-1, img_rows, img_cols, 1)
inpx = Input(shape=(img_rows, img_cols, 1))

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# Define the model using the functional API
conv1 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(inpx)
conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
bn1 = BatchNormalization()(pool1)

conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(bn1)
conv4 = Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(conv3)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
bn2 = BatchNormalization()(pool2)

conv5 = Conv2D(filters=256, kernel_size=(3, 3), activation="relu")(bn2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv5)
bn3 = BatchNormalization()(pool3)

flat = Flatten()(bn3)
dense1 = Dense(512, activation="relu")(flat)
output = Dense(10, activation="softmax")(dense1)

model = Model(inputs=inpx, outputs=output)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

model.fit(x_train, y_train, epochs=12, batch_size=500)

score = model.evaluate(x_test, y_test, verbose=0)
print('loss=', score[0])
print('acc=', score[1])
