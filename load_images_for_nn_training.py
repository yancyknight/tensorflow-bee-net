#!/usr/bin/python

#===============================================
# image_manip.py
#
# some helpful hints for those of you
# who'll do the final project in Py
#
# bugs to vladimir dot kulyukin at usu dot edu
#===============================================

import os
import cv2 # pylint: disable=locally-disabled


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
ROOT_DIR = '/home/vladimir/Desktop/nn/nn_data/nn_train/examples/'

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

######################################

