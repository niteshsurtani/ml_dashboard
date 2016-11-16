import numpy as np

from STATIC_PATH import *

def loadTrainingFile(filename):
	global DATASET_DIR

	# f = open(DATASET_DIR + filename)
	f = open(filename)
	line = f.readline().strip()
	
	values = line.split(',')

	num_instances = int(values[0])
	num_features = int(values[1])
	classes = values[2:]

	# load the CSV file as a numpy matrix
	dataset = np.loadtxt(fname = f, delimiter=",")

	# separate the data from the target attributes
	X = dataset[:,0:num_features]
	y = dataset[:,num_features]

	return X, y

def loadTestingFile(filename):
	global DATASET_DIR

	f = open(filename)
	line = f.readline().strip()

	values = line.split(',')

	num_instances = int(values[0])
	num_features = int(values[1])
	classes = values[2:]

	# load the CSV file as a numpy matrix
	dataset = np.loadtxt(fname = f, delimiter=",")

	# separate the data from the target attributes
	X = dataset[:,0:num_features]

	return X, classes