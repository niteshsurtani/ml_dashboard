import numpy as np

def loadTrainingFile(filename):
	f = open(filename)

	NUM_FEATURES = 7

	# load the CSV file as a numpy matrix
	dataset = np.loadtxt(fname = f, delimiter=",")
	# # separate the data from the target attributes
	X = dataset[:,0:NUM_FEATURES]
	y = dataset[:,NUM_FEATURES]

	return X, y

# def loadTestingFile(filename):
# 	global DATASET_DIR

# 	f = open(filename)
# 	line = f.readline().strip()

# 	values = line.split(',')

# 	num_instances = int(values[0])
# 	num_features = int(values[1])
# 	classes = values[2:]

# 	# load the CSV file as a numpy matrix
# 	dataset = np.loadtxt(fname = f, delimiter=",")

# 	# separate the data from the target attributes
# 	X = dataset[:,0:num_features]

# 	return X, classes