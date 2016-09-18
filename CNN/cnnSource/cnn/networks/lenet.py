from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense

class LeNet:
	@staticmethod
	# weightsPath: for pre-trained model
	def build(width, height, depth, classes, weightsPath=None):
		# Initialize model
		model = Sequential()

		# CONV => RELU => CONV
		model.add(Convolution2D(64, 3, 3, border_mode="same", input_shape=(depth, height, width)))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# CONV => RELU => CONV
		model.add(Convolution2D(128, 3, 3, border_mode="same"))
		model.add(Activation("relu"))
		model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

		# FC => RELU
		model.add(Flatten())
		model.add(Dense(1000))
		model.add(Activation("relu"))

		# Softmax classifier
		model.add(Dense(classes))
		model.add(Activation("softmax"))

		# If a pre-trained model exists
		if weightsPath is not None:
			model.load_weights(weightsPath)

		return model
