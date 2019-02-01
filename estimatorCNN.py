from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

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
        df_train = pd.read_csv(os.getcwd() + "\\datasets\\MNIST_train.csv", delimiter=',', header=0)
        Y_train, X_train = np.split(df_train.values, [1], axis=1)
        m = len(df_train)

        X_train = np.reshape(X_train, (m, 28, 28, 1))

        X_mean = X_train.mean().astype(np.float32)
        X_std = X_train.std().astype(np.float32)
        X_train = (X_train - X_mean) / X_std

        # data split to train / validation
        X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.1)
        return X_train.astype(np.float32), Y_train, X_valid.astype(np.float32), Y_valid

def cnn_model_fn(features, labels, mode, params):
    """Model function for CNN."""
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    if params["index"] == 1:
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu)
        conv4_flat = tf.reshape(conv4, [-1, 4 * 4 * 256])
        dense = tf.layers.dense(inputs=conv4_flat, units=4096, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=10)
    elif params["index"] == 2:
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu)
        conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu)
        conv4 = tf.layers.conv2d(inputs=conv3, filters=256, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu)
        conv4_flat = tf.reshape(conv4, [-1, 4 * 4 * 256])
        dense = tf.layers.dense(inputs=conv4_flat, units=1024, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=10)
    elif params["index"] == 3:
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu)
        conv3_flat = tf.reshape(conv3, [-1, 4 * 4 * 128])
        dense = tf.layers.dense(inputs=conv3_flat, units=4096, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=10)
    elif params["index"] == 4:
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu)
        conv3_flat = tf.reshape(conv3, [-1, 4 * 4 * 128])
        dense = tf.layers.dense(inputs=conv3_flat, units=1024, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=10)
    elif params["index"] == 5:
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[3, 3], strides=[2, 2], padding="same", activation=tf.nn.relu)
        conv3_flat = tf.reshape(conv3, [-1, 4 * 4 * 128])
        dense = tf.layers.dense(inputs=conv3_flat, units=1024, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        logits = tf.layers.dense(inputs=dense, units=10)
    elif params["index"] == 6:
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        logits = tf.layers.dense(inputs=dense, units=10)
    elif params["index"] == 7:
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.01))
        logits = tf.layers.dense(inputs=dense, units=10)
    elif params["index"] == 8:
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(0.1))
        logits = tf.layers.dense(inputs=dense, units=10)
    else:
        conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
        logits = tf.layers.dense(inputs=dropout, units=10)

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
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Load training and eval data
train_data, train_labels, eval_data, eval_labels = loadData('local')

# Train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=64, num_epochs=2, shuffle=True)

# Set up logging for predictions
#tensors_to_log = {"probabilities": "softmax_tensor"}
#logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

# train one step and display the probabilties
#mnist_classifier.train(
#    input_fn=train_input_fn,
#    steps=1,
#    hooks=[logging_hook])

models = {
    0: "2*conv_5*5, 2*pool, dense_1024, drop_0.4",
    1: "4*conv_3*3, dense_4096",
    2: "4*conv_3*3, dense_1024",
    3: "3*conv_3*3, 2*pool, dense_4096",
    4: "3*conv_3*3, 2*pool, dense_1024",
    5: "3*conv_3*3, 2*pool, dense_1024, l2reg_0.01",
    6: "2*conv_3*3, 2*pool, dense_1024",
    7: "2*conv_3*3, 2*pool, dense_1024, l2reg_0.01",
    8: "2*conv_3*3, 2*pool, dense_1024, l2reg_0.1",
}
#i = 1
for i in range(len(models)):
    mnist_classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, params={"index": i}, model_dir=os.getcwd() + "\\outputs\\model" + str(i))

    mnist_classifier.train(input_fn=train_input_fn, steps=50)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    f = open(os.getcwd() + "\\outputs\\results.txt", 'a')
    f.write('model {:2}, acc: {:<10.6}, loss: {:<10.6}, steps: {}, description: {}\n'.format(i, eval_results["accuracy"], eval_results["loss"], eval_results["global_step"], models[i]))
    print(eval_results)
    f.close()
