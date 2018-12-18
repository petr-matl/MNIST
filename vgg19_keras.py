import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.misc import imresize
import tensorflow as tf
from keras.layers import Flatten, Dense
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.utils.np_utils import to_categorical

# test preprocessing
X_test = pd.read_csv('datasets/MNIST_test.csv', delimiter=',', header=0).values
m = len(X_test)
X_test = np.reshape(X_test, (m, 28, 28))

X_resize = []
for i in range(len(X_test)):
    X_resize.append(imresize(X_test[i], (48, 48)))

X_test = np.array(X_resize)
X_test = np.stack([X_test, X_test, X_test], axis=-1)

X_mean = X_test.mean().astype(np.float32)
X_std = X_test.std().astype(np.float32)
X_test = (X_test - X_mean) / X_std

# train preprocessing
df_train = pd.read_csv('datasets/MNIST_train.csv', delimiter=',', header=0)
Y_train, X_train = np.split(df_train.values, [1], axis=1)
m = len(df_train)
classes = len(np.unique(Y_train))
Y_train = to_categorical(Y_train)

X_train = np.reshape(X_train, (m, 28, 28))

X_resize = []
for i in range(len(X_train)):
    X_resize.append(imresize(X_train[i], (48, 48)))

X_train = np.array(X_resize)
X_train = np.stack([X_train, X_train, X_train], axis=-1)  #X_train = tf.image.grayscale_to_rgb(X_train)

X_mean = X_train.mean().astype(np.float32)
X_std = X_train.std().astype(np.float32)
X_train = (X_train - X_mean) / X_std

# data split to train / validation
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1)

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(48, 48, 3))

X_input = base_model.output
X = Flatten()(X_input)
predictions = Dense(classes, activation='softmax')(X)

model = Model(inputs=base_model.input, outputs=predictions)

# First: train only the top layers (which were randomly initialized)
for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10, validation_data=(X_valid, Y_valid))

model.save_weights("models/MNIST_vgg19_model.h5")

prediction_vectors = model.predict(X_test, verbose=1)
predictions = np.argmax(prediction_vectors, axis=1)

submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions)+1)), "Label": predictions})
submissions.to_csv("outputs/MNIST_vgg19.csv", index=False, header=True)