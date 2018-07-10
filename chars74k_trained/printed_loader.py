import os
import numpy as np
import cv2
from pathlib import Path

folders = os.listdir("printed")

paths = [Path("printed/" + folder) for folder in folders]

images = []
features = []
test_images = []
test_features = []

for i in range(10):
	elements = os.listdir(paths[i])
	print(i)
	for j in range(len(elements)):
		#print(str(paths[i] / element))
		img = cv2.imread(str(paths[i]/elements[j]),cv2.IMREAD_GRAYSCALE)
		img = (255 - img)
		img = cv2.resize(img, (28,28), interpolation = cv2.INTER_AREA)
		if j < len(elements)//10 * 9:
			images.append(img)
			features.append(i)
		else:
			test_images.append(img)
			test_features.append(i)

images = np.asarray(images)
features = np.asarray(features)
test_images = np.asarray(test_images)
test_features = np.asarray(test_features)