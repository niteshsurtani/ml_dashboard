from utilities.load_file import loadTrainingFile, loadTestingFile

from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import cPickle as pickle
import sys

def kNearestNeighbour(trainingFile, MODEL_DIR, params):
	################# TRAINING ########################

	# Load Data
	X, y = loadTrainingFile(trainingFile)

	# Step 1: Import the classifier
	max_accuracy = 0.0
	best_k = 0

	for k in range(1,31):
		# Step 2: Instantiate and Estimator
		knn = KNeighborsClassifier(n_neighbors=k)

		score = cross_val_score(knn, X, y, cv=10, scoring='accuracy').mean()

		max_accuracy = max(max_accuracy,score)
		if score == max_accuracy:
			best_k = k


	# Instantiate kNN model with best parameter
	knn = KNeighborsClassifier(n_neighbors=best_k)

	# Train
	knn.fit(X, y)

	# MODEL_DIR = './models/'
	MODEL_FILE = 'knn.pkl'

	with open(MODEL_DIR + MODEL_FILE, 'wb') as dict_items_save:
		pickle.dump(knn, dict_items_save)
	return MODEL_FILE
	################# TRAINING DONE ######################

	# ################## TESTING ########################

	# with open(MODEL_DIR + MODEL_FILE, 'rb') as dict_items_open:
	# 	knn = pickle.load(dict_items_open)

	# X = loadTestingFile(testingFile)

	# # Test
	# predictions = knn.predict(X)
	
	# return MODEL_FILE, predictions

	# ################## TESTING DONE ######################

def main():
	arguments = sys.argv

	# Arguments list for Model Training

	# UserId -> fileId -> ProjectId -> ExperimentId -> ModelId -> RevisionId
	# Model dir is RevisionId

	training_file = arguments[1]
	model_dir = arguments[2]

	print training_file
	# params = []
	# if len(arguments) > 3:
	# 	params = arguments[3:]

	# MODEL_FILE = kNearestNeighbour(training_file, model_dir, params)
	# print MODEL_FILE
	

if __name__ == "__main__":
	main()
	
