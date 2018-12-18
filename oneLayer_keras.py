import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard

#np.random.seed(1)
#tf.set_random_seed(1)

X_test = pd.read_csv('datasets/MNIST_test.csv', delimiter=',', header=0).values

df_train = pd.read_csv('datasets/MNIST_train.csv', delimiter=',', header=0)
Y_train, X_train = np.split(df_train.values, [1], axis=1) # the same as X = df.values[:,1:] and Y = df.values[:,:1]
#print('X shape:', X_train.shape)

X_mean = X_train.mean().astype(np.float32)
X_std = X_train.std().astype(np.float32)
X_train = (X_train - X_mean) / X_std

m, n_x = X_train.shape
n_y = len(np.unique(Y_train))
n_layer1 = 512
batch_size = 128
num_epochs = 10
learning_rate = 0.001

Y_train = to_categorical(Y_train)
#print('Y shape', Y_train.shape)

X_input = Input(shape=(n_x,), name='input')
X = Dense(n_layer1, activation='relu', name='hidden')(X_input)
X = Dense(n_y, activation='softmax', name='output')(X)

model = Model(inputs=X_input, outputs=X, name='Neural Network')

model.compile(optimizer=Adam(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="outputs/logs/{}".format('keras'))

model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size, callbacks=[tensorboard])

prediction_vectors = model.predict(X_test)
predictions = np.argmax(prediction_vectors, axis=1)

submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions)+1)), "Label": predictions})
submissions.to_csv("outputs/MNIST.csv", index=False, header=True)