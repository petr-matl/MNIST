import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

#np.random.seed(1)
#tf.set_random_seed(1)

def one_hot(a, num_classes):
  return np.eye(num_classes)[a.reshape(-1)]

def get_minibatches(batch_size, m, X, Y):
    output_batches = []

    for index in range(0, m, batch_size):
        index_end = index + batch_size
        batch = [X[index:index_end], Y[index:index_end]]
        output_batches.append(batch)
        
    return output_batches

def dense_layer(input, channels_in, channels_out, activation=None, name="dense"):
    with tf.name_scope(name):
        initializer = tf.contrib.layers.xavier_initializer()
        w = tf.Variable(initializer([channels_in, channels_out]), name="w")
        b = tf.Variable(tf.zeros([1, channels_out]), name="b")
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)

        if (activation == 'relu'):
            a = tf.nn.relu(tf.matmul(input, w) + b)
            tf.summary.histogram("activations", a)
            return a
        else:
            z = tf.matmul(input, w) + b
            tf.summary.histogram("z", z)
            return z

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

Y_train = one_hot(Y_train, n_y)
#print('Y shape', Y_train.shape)

X = tf.placeholder(tf.float32, [None, n_x], name="X")
Y = tf.placeholder(tf.float32, [None, n_y], name="Y")

hidden = dense_layer(X, n_x, n_layer1, 'relu', "hidden")
output = dense_layer(hidden, n_layer1, n_y, "output")

with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=Y))
    tf.summary.scalar('loss', loss)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope("acc"):
    predict = tf.argmax(output, 1)
    correct_prediction = tf.equal(predict, tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('acc', accuracy)

merged_summary = tf.summary.merge_all()

minibatches = get_minibatches(batch_size, m, X_train, Y_train)

with tf.Session() as sess:
    writer = tf.summary.FileWriter("outputs/logs/{}".format('tensor'))
    writer.add_graph(sess.graph)
    
    sess.run(tf.global_variables_initializer())

    current_cost = sess.run(loss, feed_dict={X: X_train, Y: Y_train})
    train_accuracy = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
    print('Epoch: {:<4} - Loss: {:<8.3} Train Accuracy: {:<5.3} '.format(0, current_cost, train_accuracy))

    for epoch in range(num_epochs):
        epoch_cost = 0

        for minibatch in minibatches:
            minibatch_X, minibatch_Y = minibatch

            sess.run(optimizer, feed_dict={ X: minibatch_X, Y: minibatch_Y })

        s = sess.run(merged_summary, feed_dict={X: X_train, Y: Y_train})
        writer.add_summary(s, epoch)

        current_cost = sess.run(loss, feed_dict={X: X_train, Y: Y_train})
        train_accuracy = sess.run(accuracy, feed_dict={X: X_train, Y: Y_train})
        print('Epoch: {:<4} - Loss: {:<8.3} Train Accuracy: {:<5.3} '.format(epoch + 1, current_cost, train_accuracy))

    predictions = sess.run(predict, feed_dict={X: X_test})

submissions = pd.DataFrame({"ImageId": list(range(1, len(predictions)+1)), "Label": predictions})
submissions.to_csv("outputs/MNIST.csv", index=False, header=True)

#first = X[1,:].reshape(28, 28)
#print(Y[1,:])
#plt.imshow(first)
#plt.show()