import numpy as np
import math
import time
from random import randint
from NonSparsePerceptron import NonSparsePerceptron

class AdaBoostPerceptron():
	def __init__(self, T):
		self._boosting_iterations = T
		self._distribution = None
		self._alphas = []
		self._hts = []
		self._all = []
		self._training_fvs = None
		self._training_labels = None
		self._max_feature = None
		self._n = None
		self._tfinal = 0

	def train(self, training_examples, training_labels):
		self._max_feature = len(training_examples[0])
		self._training_fvs = training_examples
		self._training_labels = training_labels
		self._n = len(training_examples)
		self._alphas = []
		self._hts = []
		self._distribution = (1.0/self._n)*np.ones(self._n, dtype=float)

		for t in range(self._boosting_iterations):
			start = time.time()
			examples = []
			labels = []
			for i in range(self._n/7): # Generate random samples for perceptron
				m = randint(0, self._n - 1)
				examples.append(training_examples[m])
				labels.append(training_labels[m])

			p = NonSparsePerceptron(self._max_feature)
			p = p.train(examples, labels)
			self._all.append(p)
			
			epsilon, best_h = self.compute_error()
			self._hts.append(best_h)
			alpha = 0.5*math.log((1.0-epsilon)/epsilon)
			self._alphas.append(alpha)

			self._distribution = self.update_distribution(t)
			elapsed = time.time()
			self._tfinal += 1

		self._all = None # clear out perceptrons to save space later
		return self

	def compute_error(self):
		error_h = []
		for h in self._all:
			error = 0
			for i in range(self._n):
				fv = self._training_fvs[i]
				product = h.predict(fv)
				if product == -1:
					label = 0
				else:
					label = 1
				if label != self._training_labels[i]:
					error += self._distribution[i]

			error_h.append(error)
		return np.min(error_h), self._all[np.argmin(error_h)] # lowest error, best hypothesis for this round of training

	def predict(self, instance):
		test_feature_vector = np.zeros(self._max_feature)
		fv = instance.get_feature_vector()
		for index in fv:
			test_feature_vector[index - 1] = fv[index]

		product = 0

		for t in range(self._tfinal):
			alpha_t = self._alphas[t]
			h_t = self._hts[t]
			y_hat = h_t.predict(test_feature_vector)

			product += alpha_t*y_hat
		if product >= 0:
			output = 1
		else:
			output = 0
		return output


	def update_distribution(self,t):
		distribution = np.zeros(self._n)
		current_dist = self._distribution
		ht = self._hts[t]
		alpha = self._alphas[t]
		for i in range(self._n):
			x = self._training_fvs[i]
			val = ht.predict(x)
			y_i = self._training_labels[i]
			if y_i == 0:
				y_i = -1
			distribution[i] = current_dist[i]*math.exp(-1.0*alpha*y_i*val)

		Z = sum(distribution)
		distribution = distribution/Z
		return distribution







