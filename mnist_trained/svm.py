import mnist as m
import numpy as np
from sklearn.svm import SVC
import pickle

# Get Mnist Data
#assert(False)
X_train = []
X_test = []
Y_train = []
Y_test = []

for element in m.read("training"):
	Y_train.append(element[0])
	X_train.append(np.asarray(element[1]).flatten())

for element in m.read("testing"):
	Y_test.append(element[0])
	X_test.append(np.asarray(element[1]).flatten())

X_train = np.asarray(X_train)
X_train = X_train / 255
X_test = np.asarray(X_test)
X_test = X_test / 255
Y_train = np.asarray(Y_train)
Y_test = np.asarray(Y_test)

# Prepare SVM

svm_model = SVC(probability = False, kernel = "rbf", C = 2.8, gamma = .0073, cache_size = 10000) # Since we have 16GB of RAM I will use 10GB as cache for this.

print("Inizio fitting. Richiederà un bel po' di tempo come cosa...")
svm_model.fit(X_train, Y_train)

# save the model to disk
filename = 'svm_mnist.sav'
pickle.dump(svm_model, open(filename, 'wb'))

# Testing 
# Get confusion matrix
from sklearn import metrics
predicted = clf.predict(X_test)
print("Confusion matrix:\n%s" %
      metrics.confusion_matrix(Y_test,
                               predicted))
print("Accuracy: %0.4f" % metrics.accuracy_score(Y_test,
                                                 predicted))

