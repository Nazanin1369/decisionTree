import sys
from class_vis import prettyPicture
from prep_terrain_data import makeTerrainData
from sklearn import tree
from sklearn.metrics import accuracy_score

import numpy as np
import pylab as pl
features_train, labels_train, features_test, labels_test = makeTerrainData()

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)

acc = submitAccuracies(accuracy_score(labels_test, clf.predict([labels_test])))

def submitAccuracies():
  return {"acc":round(acc,3)}

