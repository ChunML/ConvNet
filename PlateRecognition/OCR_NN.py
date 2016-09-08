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
		
		self.test_data = np.loadtxt('test.csv')
		self.test_data = self.test_data/255.

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

	# Sigmoid function's Derivative
	def sigmoid_derivative(self, z):
		return self.sigmoid(z)*(1 - self.sigmoid(z))

	# Vectorize output y to a (m x 1) vector
	def vectorize(self, a):
		y_vectorized = np.zeros((len(self.biases[-1]), 1))
		y_vectorized[a] = 1.
		return y_vectorized

	# Back propagation
	def back_propagation(self):
		for mini_batch in self.mini_batches:
			nabla_w = [np.zeros(w.shape) for w in self.weights]
			nabla_b = [np.zeros(b.shape) for b in self.biases]
			for X, y in mini_batch:
				
				self.computeActivations(X)

				y_vectorized = self.vectorize(int(y))

				# Compute delta of the last layer
				delta_last = -(y_vectorized - self.activations[-1])*self.sigmoid_derivative(self.zs[-1])

				single_nabla_w, single_nabla_b = self.calculateNabla(delta_last)
				
				# Compute gradient
				nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, single_nabla_w)]
				nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, single_nabla_b)]

			temp_weights = self.weights
			temp_biases = self.biases

			# Update weights and biases of the network
			self.weights = [w - (3. / len(mini_batch))*nw for w, nw in zip(temp_weights, nabla_w)]
			self.biases = [b - (3. / len(mini_batch))*nb for b, nb in zip(temp_biases, nabla_b)]

	# Compute nabla_w, nabla_b for each X, y in training set
	def calculateNabla(self, delta):

		nabla_w = [np.zeros(w.shape) for w in self.weights]
		nabla_b = [np.zeros(b.shape) for b in self.biases]

		nabla_w[-1] = np.dot(delta, self.activations[-2].transpose()).transpose()
		nabla_b[-1] = delta

		for l in range(2, self.num_layers):
			delta = [a * b for a, b in zip(np.dot(self.weights[-l+1], delta), self.sigmoid_derivative(self.zs[-l]))]
			nabla_b[-l] = delta
			nabla_w[-l] = np.dot(delta, self.activations[-l - 1].transpose()).transpose()

		return nabla_w, nabla_b

	# Execute function
	def runNetwork(self, epochs, mini_batch_size):
		for i in range(epochs):
			self.mini_batches = [list(zip(self.train_data[j:j+mini_batch_size], self.train_target[j:j+mini_batch_size])) for j in range(0, len(self.train_data), mini_batch_size)]
			self.back_propagation()
			sum = 0;
			for X, y in zip(self.test_data, self.test_target):
				self.computeActivations(X)
				sum += int(np.argmax(self.activations[-1]) == y)
			print('Epoch {}: {}/{}'.format(i, sum, len(self.test_target)))

	def test(self):
		results = []
		for X in self.test_data:
			self.computeActivations(X)
			print(np.argmax(self.activations[-1]))
			results.append(np.argmax(self.activations[-1]))
		return results