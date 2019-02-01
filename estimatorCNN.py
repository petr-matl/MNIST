from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

tf.logging.set_verbosity(tf.logging.INFO)

def loadData(source):
    if (source == 'keras'):
        ((train_data, train_labels),
        (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

        train_data = train_data/np.float32(255)
        train_labels = train_labels.astype(np.int32)  # not required

        eval_data = eval_data/np.float32(255)
        eval_labels = eval_labels.astype(np.int32)  # not required

        return train_data, train_labels, eval_data, eval_labels
    else:
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

        X_train = np.reshape(X_train, (m, 28, 28, 1))

        X_mean = X_train.mean().astype(np.float32)
        X_std = X_train.std().astype(np.float32)
        X_train = (X_train - X_mean) / X_std

        # data split to train / validation
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1)
        return X_train, Y_train, X_valid, Y_valid

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    #conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    #pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    #conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    #pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    #pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    #dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    #dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    #logits = tf.layers.dense(inputs=dropout, units=10)

    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    conv4_flat = tf.reshape(conv4, [-1, 28 * 28 * 256])
    dense = tf.layers.dense(inputs=conv4_flat, units=4096, activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=dense, units=10)

    #hidden = tf.layers.conv2d(inputs=X, filters=32, kernel_size=[3, 3], strides=[1, 1], padding="same", activation=tf.nn.relu, name='conv1')
    #hidden = tf.layers.conv2d(inputs=hidden, filters=64, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu,  name='conv2')
    #hidden = tf.layers.conv2d(inputs=hidden, filters=128, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu, name='conv3')
    #hidden = tf.layers.conv2d(inputs=hidden, filters=256, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu, name='conv4')
    #hidden = tf.contrib.layers.flatten(hidden)
    #hidden = tf.layers.dense(hidden, 4096, activation=tf.nn.relu, name='dense1')
    #output = tf.layers.dense(hidden, 10, activation=tf.nn.softmax, name='dense2')

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Load training and eval data
train_data, train_labels, eval_data, eval_labels = loadData('local')

# Create the Estimator
mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="c:/Users/matl/Disk Google/Projects/MNIST/outputs/model")

# Set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}

logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_data},
    y=train_labels,
    batch_size=64,
    num_epochs=2,
    shuffle=True)

# train one step and display the probabilties
#mnist_classifier.train(
#    input_fn=train_input_fn,
#    steps=1,
#    hooks=[logging_hook])

mnist_classifier.train(input_fn=train_input_fn)#, steps=1000)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": eval_data},
    y=eval_labels,
    num_epochs=1,
    shuffle=False)

eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)