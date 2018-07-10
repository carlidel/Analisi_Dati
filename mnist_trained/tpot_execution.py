import mnist as m
import numpy as np
from tpot import TPOTClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import cv2

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

# Fun Fact: I wanted so hard 2D data in numpy but TPot wants only 1D data :^(
print(X_train[1])
print(Y_train[1])
tpot = TPOTClassifier(generations=5, population_size=20, verbosity=3)
tpot.fit(X_train, Y_train)
tpot.export('tpot_mnist_pipeline.py')
print(tpot.score(X_test, Y_test))
