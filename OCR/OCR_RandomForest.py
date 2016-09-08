from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn import metrics
import numpy as np
import random
import matplotlib.pyplot as plt

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
	params = {'n_estimators': [10, 30, 50], 'criterion': ('gini', 'entropy'), 'verbose': [0, 1, 2]} #, 'max_depth': [5, 10], 'verbose': [0, 1, 2]}
	
	clf = RandomForestClassifier()
	grid = GridSearchCV(clf, params, n_jobs=-1)
	return grid

def train_evaluate(clf):
	data, target = load_data()
	clf.fit(data[:60000], target[:60000])
	return clf

def run():
	clf = initialize()
	clf = train_evaluate(clf)
	data, target = load_data()
	result = clf.predict(data[60000:])

	# Best result: 99.49%
	# Params: criterion: 'gini', n_estimators=50, verbose=1
	print(clf.best_estimator_)
	print(np.mean(metrics.accuracy_score(target[60000:], result)))