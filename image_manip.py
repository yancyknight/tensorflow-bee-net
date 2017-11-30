#!/usr/bin/python

#===============================================
# image_manip.py
#
# some helpful hints for those of you
# who'll do the final project in Py
#
# bugs to vladimir dot kulyukin at usu dot edu
#===============================================

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required = True, help = 'Path to image')
args = vars(ap.parse_args())

## load an image from a path specified in args['image']
image = cv2.imread(args['image'])
print image

## display the image for the user
cv2.imshow('Loaded Image', image)

## read and normalize the image
normalized_image = (cv2.imread(args['image'])/float(255))
print normalized_image

cv2.waitKey(0)
