import numpy as np

class Perceptron:
	def __init__(self):
		self._w = None 
		self._learning_rate = 1.0
		self._iterations = 5

	def train(self, instances):
		I = self._iterations
		eta = self._learning_rate
		w = {}
		for k in range(I):
			for i in range(len(instances)):
				product = 0
				x = instances[i].get_feature_vector() # ith example as dictionary fv
				for index in x:
					try:
						product += w[index]*x[index]
					except KeyError:
						w[index] = 0 #initialize w for the missing index

				y_hat = 0
				if product >=  0:
					y_hat = 1
				else:
					y_hat = -1

				#update
				correct_label = instances[i].get_label()
				if correct_label == 0:
					correct_label = -1
				if y_hat != correct_label:
					for index in x:
						w[index] += eta*correct_label*x[index]

		self._w = w

		return self

	def predict(self, instance):
		w = self._w
		x = instance.get_feature_vector()# instance feature vector. Will not have more features than training set.
		for index in x:
			try:
				product += w[index]*x[index]
			except KeyError:
				product += 0 # Non-training features
		
		if product >= 0:
			output = 1
		else:
			output = 0
		return output


