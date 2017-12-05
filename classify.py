#!/usr/bin/python

#===============================================
# 
#       READ IMAGES
# 
#===============================================

from __future__ import division
# import time
import os
import cv2 
import tensorflow as tf
import numpy as np


# two dictionaries that map integers to images, i.e.,
# 2D numpy array.
TRAIN_IMAGE_DATA = {}
TEST_IMAGE_DATA  = {}

# the train target is an array of 1's
TRAIN_TARGET = []
# the set target is an array of 0's.
TEST_TARGET  = []

### Global counters for train and test samples
NUM_TRAIN_SAMPLES = 0
NUM_TEST_SAMPLES  = 0

## define the root directory
ROOT_DIR = '/home/yancy/Documents/computability/project/data/nn_dev/'

## read the single bee train images
YES_BEE_TRAIN = ROOT_DIR + 'single_bee_train'

for root, dirs, files in os.walk(YES_BEE_TRAIN):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TRAIN_IMAGE_DATA[NUM_TRAIN_SAMPLES] = img
            TRAIN_TARGET.append(int(1))
        NUM_TRAIN_SAMPLES +=1


## read the single bee test images
YES_BEE_TEST = ROOT_DIR + 'single_bee_test'

for root, dirs, files in os.walk(YES_BEE_TEST):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            # print img.shape
            TEST_IMAGE_DATA[NUM_TEST_SAMPLES] = img
            TEST_TARGET.append(int(1))
        NUM_TEST_SAMPLES += 1

## read the no-bee train images
NO_BEE_TRAIN = ROOT_DIR + 'no_bee_train'

for root, dirs, files in os.walk(NO_BEE_TRAIN):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TRAIN_IMAGE_DATA[NUM_TRAIN_SAMPLES] = img
            TRAIN_TARGET.append(int(0))
        NUM_TRAIN_SAMPLES += 1
        
# read the no-bee test images
NO_BEE_TEST = ROOT_DIR + 'no_bee_test'

for root, dirs, files in os.walk(NO_BEE_TEST):
    for item in files:
        if item.endswith('.png'):
            ip = os.path.join(root, item)
            img = (cv2.imread(ip)/float(255))
            TEST_IMAGE_DATA[NUM_TEST_SAMPLES] = img
            TEST_TARGET.append(int(0))
        NUM_TEST_SAMPLES += 1

print NUM_TRAIN_SAMPLES
print NUM_TEST_SAMPLES
TRAIN_IMAGE_CLASSIFICATIONS = zip([k for k in TRAIN_IMAGE_DATA.keys()], TRAIN_TARGET)
TEST_IMAGE_CLASSIFICATIONS = zip([k for k in TEST_IMAGE_DATA.keys()], TEST_TARGET)
print TRAIN_IMAGE_CLASSIFICATIONS
print TEST_IMAGE_CLASSIFICATIONS

#===============================================
# 
#       CLASSIFY IMAGES
# 
#===============================================

# written following a tutorial at https://www.tensorflow.org/tutorials/layers

def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    # Logits Layer
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
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)

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
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

eval_metric_ops = {
    "accuracy": tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"])}
return tf.estimator.EstimatorSpec(
    mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def main(unused_argv):
    # Load training and eval data
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = TRAIN_IMAGE_DATA
    train_labels = TRAIN_IMAGE_CLASSIFICATIONS
    eval_data = TEST_IMAGE_CLASSIFICATIONS
    eval_labels = TEST_IMAGE_CLASSIFICATIONS

mnist_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

if __name__ == '__main__':
    tf.app.run()


