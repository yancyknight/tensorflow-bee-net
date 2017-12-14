#!/usr/bin/python

from __future__ import division
import os
import cv2 
import tensorflow as tf
import numpy as np

def testNet(netpath, dirpath):
    TEST_IMAGE_DATA = []

    for root, dirs, files in os.walk(dirpath):
        for item in files:
            if item.endswith('.png'):
                ip = os.path.join(root, item)
                img = (cv2.imread(ip)/float(255))
                TEST_IMAGE_DATA.append(img)

    for image in TEST_IMAGE_DATA:
        eval_data = np.asarray(image)

        session = tf.Session()

        saver = tf.train.import_meta_graph(netpath + '/model.ckpt-20001.meta')
        saver.restore(session, tf.train.latest_checkpoint(netpath))

        graph = tf.get_default_graph()
        op_to_restore = graph.get_operation_by_name('softmax_tensor')
        print session.run(op_to_restore, eval_data)

testNet('./bee_convnet_model','./data/nn_dev/no_bee_test')
# testNet('./bee_convnet_model','./data/nn_dev/single_bee_test')
