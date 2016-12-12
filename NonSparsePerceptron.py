import numpy as np

class NonSparsePerceptron: # returns true labels (1 and -1)
	def __init__(self, max_feature):
		self._w = None 
		self._learning_rate = 1.0
		self._iterations = 2
		self._max_feature = max_feature

	def train(self, training_examples, training_labels):
		I = self._iterations
		eta = self._learning_rate
		w = np.zeros(self._max_feature)
		for k in range(I):
			for i in range(len(training_examples)):
				x = training_examples[i]
				product = w.dot(x)

				y_hat = 0
				if product >=  0:
					y_hat = 1
				else:
					y_hat = -1

				#update
				correct_label = training_labels[i]
				if correct_label == 0:
					correct_label = -1
				if y_hat != correct_label:
					w += eta*correct_label*x

		self._w = w

		return self

	def predict(self, example):
		w = self._w
		x = example # instance feature vector. Will not have more features than training set.
		product = w.dot(x)
		
		if product >= 0:
			output = 1
		else:
			output = -1

		return output

	def get_weight_vector(self):
		return self._w


