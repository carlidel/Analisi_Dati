import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
import cv2
import mnist as m
import numpy as np

# NOTE: Make sure that the class is labeled 'target' in the data file
X_train = []
X_test = []
Y_train = []
Y_test = []

for element in m.read("training"):
	Y_train.append(element[0])
	X_train.append(cv2.resize(np.asarray(element[1]),(14,14), interpolation = cv2.INTER_AREA).flatten())

for element in m.read("testing"):
	Y_test.append(element[0])
	X_test.append(cv2.resize(np.asarray(element[1]),(14,14), interpolation = cv2.INTER_AREA).flatten())


# Score on the training set was:0.9651667280513239
exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="gini", max_features=0.15000000000000002, min_samples_leaf=1, min_samples_split=18, n_estimators=100)

exported_pipeline.fit(X_train, Y_train)
# save the model to disk
filename = 'tpot_model.sav'
pickle.dump(exported_pipeline, open(filename, 'wb'))
