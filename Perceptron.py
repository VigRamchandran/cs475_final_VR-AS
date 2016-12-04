import numpy as np

class Perceptron:
	def __init__(self):
		self._w = None 
		self._learning_rate = 1.0
		self._iterations = 5

	def train(self, instances, training_labels):
		I = self._iterations
		eta = self._learning_rate
		w = np.zeros(len(instances[0]))
		for k in range(I):
			for i in range(len(instances)):
				x = np.array(instances[i]) # ith example as np array
				product = w.dot(x) # take dot product

				y_hat = 0
				if product >=  0:
					y_hat = 1
				else:
					y_hat = -1

				#update
				correct_label = training_labels[i]
				if y_hat != correct_label:
					w +=  eta*correct_label*x

		self._w = w

		return self

	def predict(self, instance):
		w = self._w
		x = np.array(instance) # instance feature vector. Will not have more features than training set.
		product = w.dot(x)
		
		if product >= 0:
			output = 1
		else:
			output = 0
		return output


