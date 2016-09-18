from cnnSource.cnn.networks import lenet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2

# Construct argument parse
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1)
ap.add_argument("-l", "--load-model", type=int, default=-1)
ap.add_argument("-w", "--weights", type=str)
args = vars(ap.parse_args())

# Load CIFAR-10 datasets
# Flatten CIFAR-10 data and rescale to [0 1.0] range
print("[INFO] loading CIFAR-10...")

data = np.load('cifar-10/data.npy')
data = data.reshape(data.shape[0], 3, 32, 32)

train_data, test_data, train_label, test_label = train_test_split(data / 255., np.load('cifar-10/label.npy').astype('int'), test_size=0.15)

# Vectorize label
# Ex: y = '2' -> y = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
train_label = np_utils.to_categorical(train_label, 10)
test_label = np_utils.to_categorical(test_label, 10)

# Initialize the optimizer and model
print("[INFO] compiling model...")
opt = SGD(lr=0.01) # learning rate = 0.01
model = lenet.LeNet.build(width=32, height=32, depth=3, classes=10, weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# Train if there is no pre-trained model
if args["load_model"] < 0:
        print("[INFO] training...")
        model.fit(train_data, train_label, batch_size=128, nb_epoch=50, verbose=1)

        print("[INFO] evaluating...")
        loss, accuracy = model.evaluate(test_data, test_label, batch_size=128, verbose=1)
        print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

if args["save_model"] > 0:
        print("[INFO] dumping weights to file...")
        model.save_weights(args["weights"], overwrite=True)

# Randomly print test image with predicted label
for i in np.random.choice(np.arange(0, len(test_label)), size=(30,)):
        # Classify the digit
        probs = model.predict(test_data[np.newaxis, i])
        pred = probs.argmax(axis=1)

        # Enlarge the image
        image = cv2.resize(test_data[i][0], (96, 96), interpolation=cv2.INTER_LINEAR)
        cv2.putText(image, str(pred[0]), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        print("[INFO] Predicted: {}, Actual: {}".format(pred[0], np.argmax(test_label[i])))
        cv2.imshow("Digit", image)
        cv2.waitKey(0)
