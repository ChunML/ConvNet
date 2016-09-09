from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import numpy as np
import random
import matplotlib.pyplot as plt
import cv2

def load_data():
	# Fetch MNIST Dataset
	mnist = datasets.fetch_mldata('MNIST Original')
	data, target = mnist.data / 255., mnist.target

	# Shuffle the dataset
	random_indexes = [i for i in range(70000)]
	random.shuffle(random_indexes)
	data = [data[ri] for ri in random_indexes]
	target = [target[ri] for ri in random_indexes]
	return data, target

def initialize():
	return RandomForestClassifier(n_estimators=50, criterion='gini', verbose=1)

def train_evaluate(clf):
	data, target = load_data()
	clf.fit(data[:60000], target[:60000])
	return clf