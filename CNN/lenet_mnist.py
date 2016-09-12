from cnnSource.cnn.networks import lenet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
#import cv2

# Construct argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1)
ap.add_argument("-l", "--load-model", type=int, default=-1)
ap.add_argument("-w", "--weights", type=str)
args = vars(ap.parse_args())

# Download MNIST datasets
print("[INFO] downloading MNIST...")
dataset = datasets.fetch_mldata("MNIST Original")

# Flatten MNIST data and rescale to [0 1.0] range
data = dataset.data.reshape((dataset.data.shape[0], 28, 28))
data = data[:, np.newaxis, :, :]
train_data, test_data, train_label, test_label = train_test_split(data / 255., dataset.target.astype('int'), test_size=0.33)

# Vectorize label 
# Ex: y = '2' -> y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
train_label = np_utils.to_categorical(train_label, 10)
test_label = np_utils.to_categorical(test_label, 10)

# Initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01) # learning rate = 0.01
model = lenet.LeNet.build(width=28, height=28, depth=1, classes=10, weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=None, metrics=["accuracy"])

# Train if there is no pre-trained model
if args["load_model"] < 0:
	print("[INFO] training...")
	model.fit(train_data, train_label, batch_size=128, nb_epoch=20, verbose=1)

	print("[INFO] evaluating...")
	loss, accuracy = model.evaluate(test_data, test_label, batch_size=128, verbose=1)
	print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)