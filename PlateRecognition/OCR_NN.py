from sklearn import datasets
from sklearn.cross_validation import train_test_split
import numpy as np
import random

# Network Class
class Network:

	# Class constructor
	def __init__(self, networkSizes):
		self.num_layers = len(networkSizes)
		# self.weights = [np.random.randn(x, y) for x, y in zip(networkSizes[:-1], networkSizes[1:])]
		# Initialize weights with standard deviation of 1/sqrt(input_size)
		self.weights = [np.loadtxt('weights_0.csv'), np.loadtxt('weights_1.csv')]
		self.biases = [np.loadtxt('biases_0.csv'), np.loadtxt('biases_1.csv')]

	# Compute the activations of all layers
	def computeActivations(self, X):
		self.activations = []
		self.zs = []
		self.activations = [np.reshape(X, (len(X), 1))]
		for w, b in zip(self.weights, self.biases):
			b = np.reshape(b, (len(b), 1))
			z = np.dot(w.transpose(), self.activations[-1]) + b
			z = np.array(z)
			a = self.sigmoid(z)
			self.zs.append(z)
			self.activations.append(a)

	# Sigmoid function
	def sigmoid(self, z):
		return 1.0 / (1.0 + np.exp(-z))

	def predict(self, test_data):
		results = []
		for X in test_data:
			self.computeActivations(X)
			results.append(np.argmax(self.activations[-1]))
		return results