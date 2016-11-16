# Load dataset
# K-Fold cross validation
# Calculate ratio
# Combine result for complete data 


import numpy as np
from sklearn.svm import SVR
from sklearn.cross_validation import KFold
from load_file import loadTrainingFile
import sys
from sklearn.svm import NuSVR
import matplotlib.pyplot as plt
from sklearn import tree

def SVM(trainingFile):
	# Load Data
	X, y = loadTrainingFile(trainingFile)
	num_instances = len(y)
	num_folds = 10

	kf = KFold(num_instances, n_folds=num_folds, shuffle=True)

	player = {}
	salary = []
	ratio = []

	for train_index, test_index in kf:

		X_train, X_test = X[train_index], X[test_index]
		y_train, y_test = y[train_index], y[test_index]

		clf = tree.DecisionTreeRegressor()
		clf.fit(X_train, y_train)


		# svr_rbf = SVR(kernel='rbf', C=1, gamma=0.1)
		# model = svr_rbf.fit(X_train, y_train)

		y_rbf = clf.predict(X_test)

		for i in range(0, len(test_index)):
			player[test_index[i]] = y_rbf[i] * 1.0 / y_test[i] 
			salary.append(y_test[i])
			ratio.append(y_rbf[i] * 1.0 / y_test[i])

		# print len(X_test), len(y_test), len(y_rbf)

		plt.scatter(salary, ratio, c='k', label='data')
		# plt.hold('on')
		# plt.plot(X_test, y_rbf, c='g', label='RBF model')
		plt.xlabel('salary')
		plt.ylabel('Value')
		plt.title('Undervalued - Overvalued')
		plt.legend()
		plt.show()
		
	return player

# ###############################################################################
# # Generate sample data
# X = np.sort(5 * np.random.rand(40, 1), axis=0)
# y = np.sin(X).ravel()

# ###############################################################################
# # Add noise to targets
# y[::5] += 3 * (0.5 - np.random.rand(8))

# ###############################################################################
# # Fit regression model
# svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
# svr_lin = SVR(kernel='linear', C=1e3)
# svr_poly = SVR(kernel='poly', C=1e3, degree=2)
# y_rbf = svr_rbf.fit(X, y).predict(X)
# y_lin = svr_lin.fit(X, y).predict(X)
# y_poly = svr_poly.fit(X, y).predict(X)

# ###############################################################################
# print "Here"
# # look at the results
# plt.scatter(X, y, c='k', label='data')
# plt.hold('on')
# plt.plot(X, y_rbf, c='g', label='RBF model')
# plt.plot(X, y_lin, c='r', label='Linear model')
# plt.plot(X, y_poly, c='b', label='Polynomial model')
# plt.xlabel('data')
# plt.ylabel('target')
# plt.title('Support Vector Regression')
# plt.legend()
# plt.show()


def main():
	arguments = sys.argv

	training_file = arguments[1]
	SVM(training_file)
	

if __name__ == "__main__":
	main()
	
