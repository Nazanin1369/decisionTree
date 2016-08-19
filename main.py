#!/usr/bin/python

import sys
from class_vis import prettyPicture, output_image
from prep_terrain_data import makeTerrainData

import matplotlib.pyplot as plt
import numpy as np
import pylab as pl
from classifier import classify

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the classify() function in classifyDT is where the magic
clf = classify(features_train, labels_train)



prettyPicture(clf, features_test, labels_test)
output_image("test.png", "png", open("test.png", "rb").read())
