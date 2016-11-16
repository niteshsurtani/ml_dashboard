def convertToLabel(predictions, classLabel):
	prediction_labels = []
	for output in predictions:
		prediction_labels.append(classLabel[int(output)])

	return prediction_labels