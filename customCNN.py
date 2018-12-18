import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

def one_hot(a, num_classes):
  return np.eye(num_classes)[a.reshape(-1)]

def get_minibatches(batch_size, m, X, Y):
    output_batches = []

    for index in range(0, m, batch_size):
        index_end = index + batch_size
        batch = [X[index:index_end], Y[index:index_end]]
        output_batches.append(batch)
        
    return output_batches

# test preprocessing
X_test = pd.read_csv('datasets/MNIST_test.csv', delimiter=',', header=0).values
m = len(X_test)
X_test = np.reshape(X_test, (m, 28, 28, 1))

X_mean = X_test.mean().astype(np.float32)
X_std = X_test.std().astype(np.float32)
X_test = (X_test - X_mean) / X_std

# train preprocessing
df_train = pd.read_csv('datasets/MNIST_train.csv', delimiter=',', header=0)
Y_train, X_train = np.split(df_train.values, [1], axis=1)
m = len(df_train)
classes = len(np.unique(Y_train))
Y_train = one_hot(Y_train, classes)

m = 1000
X_train = X_train[:m,:]
Y_train = Y_train[:m,:]


X_train = np.reshape(X_train, (m, 28, 28, 1))

X_mean = X_train.mean().astype(np.float32)
X_std = X_train.std().astype(np.float32)
X_train = (X_train - X_mean) / X_std

# data split to train / validation
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1)

X = tf.placeholder(tf.float32, [None, 28, 28, 1], name="X")
Y = tf.placeholder(tf.float32, [None, classes], name="Y")

hidden = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same", activation=tf.nn.relu, name='conv1')
hidden = tf.layers.conv2d(inputs=hidden, filters=64, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu,  name='conv2')
hidden = tf.layers.conv2d(inputs=hidden, filters=128, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu, name='conv3')
hidden = tf.layers.conv2d(inputs=hidden, filters=256, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu, name='conv4')
hidden = tf.contrib.layers.flatten(hidden)
hidden = tf.layers.dense(hidden, 4096, activation=tf.nn.relu, name='dense1')
output = tf.layers.dense(hidden, 10, activation=tf.nn.softmax, name='dense2')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))

optimizer = tf.train.AdamOptimizer().minimize(loss)

predict = tf.argmax(output, 1)
correct_prediction = tf.equal(predict, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

minibatches = get_minibatches(64, X_train.shape[0], X_train, Y_train)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    #current_cost = sess.run(loss, feed_dict={X: X_train, Y: Y_train})
    #train_accuracy = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
    #print('Epoch: {:<4} - Loss: {:<8.3} Train Accuracy: {:<5.3} '.format(0, current_cost, train_accuracy))

    for epoch in range(10):
        epoch_cost = 0

        minibatch_index = 0
        for minibatch in minibatches:
            minibatch_X, minibatch_Y = minibatch
            
            sess.run(optimizer, feed_dict={ X: minibatch_X, Y: minibatch_Y })

            minibatch_index += 1
            if minibatch_index % 10 == 0:
                print("Current minibatch: {:d} / {:d}".format(minibatch_index, len(minibatches)), end="\r")

        current_cost = sess.run(loss, feed_dict={X: X_train, Y: Y_train})
        train_accuracy = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
        print('Epoch: {:<4} - Loss: {:<8.3} Train Accuracy: {:<5.3} '.format(epoch + 1, current_cost, train_accuracy))

    predictions = sess.run(predict, feed_dict={X: X_test})

submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions)+1)), "Label": np.argmax(predictions)})
submissions.to_csv("outputs/MNIST.csv", index=False, header=True)