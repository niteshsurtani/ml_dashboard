from utilities.load_file import loadTrainingFile, loadTestingFile
from utilities.convert_label import convertToLabel

from sklearn.externals import joblib
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier

import cPickle as pickle
import sys


def kNNTest(testingFile, MODEL_DIR):

	# MODEL_DIR = './models/'
	MODEL_FILE = 'knn.pkl'

	################## TESTING ########################

	with open(MODEL_DIR + MODEL_FILE, 'rb') as dict_items_open:
		knn = pickle.load(dict_items_open)

	X, classLabel = loadTestingFile(testingFile)

	# Test
	predictions = knn.predict(X)
	
	prediction_labels = convertToLabel(predictions, classLabel)
	return prediction_labels

	################## TESTING DONE ######################

def main():
	arguments = sys.argv
	test_file = arguments[1]
	model_dir = arguments[2]


	predictions = kNNTest(test_file, model_dir)
	print predictions

if __name__ == "__main__":
	main()
	
