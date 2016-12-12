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
			print "On boosting iteration: {}".format(t) ,
			examples = []
			labels = []
			for i in range(len(instances)/10): # Generate random samples for perceptron
				m = randint(0, len(training_examples) - 1)
				examples.append(training_examples[m])
				labels.append(training_labels[m])

			p = NonSparsePerceptron(self._max_feature)
			p = p.train(training_examples, training_labels)
			self._all.append(p)
			
			epsilon, best_h = compute_error()
			self._hts.append(best_h)
			alpha = 0.5*math.log((1.0-epsilon)/epsilon)
			self._alphas.append(alpha)

			self._distribution = self.update_distribution()

			print "\t {} (s)".format(elapsed-start)
		return self

	def compute_error():
		for h in self._all:
			error = 0
			for i in range(self._n):
				fv = self._training_fvs[i]
				product = h.predict(fv)
				if product >= 0:
					label = 1
				else:
					label = 0
				if label != self._training_labels[i]:
					error += self._distribution[i]

			error_h.append(error)
		return np.min(error_h), np.argmin(error_h) # lowest error, best hypothesis for this round of training

	def predict(self, testing_example):
		test_feature_vector = testing_example
		product = 0

		for t in range(self._tfinal):
			alpha_t = self._alphas[t]
			j_t = self._j[t]
			c_t = self._c[t]
			h_t = self._hts[t]
			if test_feature_vector[j_t] > c_t:
				y_hat = h_t[0]
			else:
				y_hat = h_t[1]

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
			distribution[i] = current_dist[i]*math.exp(-1.0*alpha*y_i*val)

		Z = sum(distribution)
		distribution = distribution/Z
		return distribution







