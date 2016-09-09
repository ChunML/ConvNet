from sklearn import datasets
import cv2
import numpy as np

def load_data():
	# Fetch MNIST Dataset
	mnist = datasets.fetch_mldata('MNIST Original')
	data, target = mnist.data / 255., mnist.target
	return data, target

data, target = load_data()
test = np.reshape(data[50000], (28, 28))
cv2.imshow("",test)

cv2.waitKey(0)
cv2.destroyAllWindows()