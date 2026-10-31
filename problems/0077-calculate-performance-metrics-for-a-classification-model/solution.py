
def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
	# Implement your code here

	tp, tn, fp, fn = 0,0,0,0
	
	# tp, tn, fp, fn
	for i in range(len(actual)):
		if actual[i] == predicted[i] and predicted[i] == 1:
			tp += 1
	
	for i in range(len(actual)):
		if actual[i] == predicted[i] and predicted[i] == 0:
			tn += 1

	for i in range(len(actual)):
		if actual[i] != predicted[i] and actual[i] == 0:
			fp += 1
	
	for i in range(len(actual)):
		if actual[i] != predicted[i] and actual[i] == 1:
			fn += 1
	

	accuracy = (tp + tn) / (tp+tn+fp+fn)

	precision = tp / (tp+fp)
	recall = tp / (tp + fn)

	f1 = (2 * precision * recall) / (precision + recall)

	specificity = tn / (tn+fp)
	negativePredictive = (tn) / (tn+fn)

	confusion_matrix = [[tp, fn], [fp, tn]]

	return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)
