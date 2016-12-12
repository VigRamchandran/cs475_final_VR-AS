#Vignesh Ramchandran
#Machine Learning HW 3
import numpy as np

class KNN():
	def __init__(self, k):
		self._k = 5
		self._training_fvs = None
		self._training_labels = None
		self._max_feature = 1
		self._labels = None

	def train(self, training_examples, training_labels):

		# Initialize some stuff
		self._max_feature = len(training_examples[0])
		self._training_fvs = training_examples
		self._training_labels = training_labels
		self._labels = list(set(training_labels))

		return self
		

	def predict(self, instance):
		test_feature_vector = np.zeros(self._max_feature)
		fv = instance.get_feature_vector()
		for index in fv:
			test_feature_vector[index - 1] = fv[index]

		indicies = self.k_nearest_neighbors(self._k, self._training_fvs, self._training_labels, test_feature_vector)[0]

		votes = np.zeros(max(self._labels)+1)

		for i in indicies:
			label = self._training_labels[i]
			votes[label] += 1
		q = np.argmax(votes)

		return q


	def k_nearest_neighbors(self, k, training_examples, training_labels, testing_example):
		counter = 0
		values = []
		for training_example in training_examples:
			distance = np.linalg.norm(training_example - testing_example)
			values.append((distance, counter, training_labels[counter]))
			counter += 1


		dtype = [("distance", float), ("index", int), ("label", int)]
		k_nearest_neighbors = np.sort(np.array(values, dtype=dtype), order=["distance","label"])[0:k]
			
		k_closest_indicies = []
		k_smallest_differences = []

		for tup in k_nearest_neighbors:
			k_smallest_differences.append(tup[0])
			k_closest_indicies.append(tup[1])

		return (k_closest_indicies, k_smallest_differences)
