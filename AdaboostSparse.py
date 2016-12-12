import numpy as np
import math
import time

class AdaboostSparse():
	def __init__(self, T):
		self._boosting_iterations = T
		self._distribution = None
		self._alphas = []
		self._hts = []
		self._j = []
		self._c = []
		self._tfinal = None
		self._max_feature = None
		self._n = None
		self._tfinal = 1
	def train(self, instances):
		curr_max = 1
		for instance in instances:
			if instance.get_max_feature() > curr_max:
				curr_max = instance.get_max_feature()

		self._max_feature = curr_max
		self._instances = instances

		self._n = len(instances)
		self._alphas = []
		self._hts = []
		self._j = []
		self._c = []
		self._distribution = (1.0/self._n)*np.ones(self._n, dtype=float)

		print "Computing cutoffs ..."
		cutoffs = self.compute_c(instances)
		print "Computing hypothesis classes ..."
		hjc = self.compute_hjc_all(cutoffs)

		for t in range(self._boosting_iterations):
			start = time.time()
			print "On boosting iteration: {}".format(t) ,
			errors = self.compute_errors(hjc, cutoffs)
			min_error = self.min_error(errors)
			best_j = min_error[1]
			best_c = min_error[2]
			epsilon = min_error[0]
			if epsilon < 0.000001:
				if t == 0: #for vision when this occurs
					self._alphas.append(1)
					self._hts.append(hjc[best_j][best_c])
					self._j.append(best_j)
					self._c.append(cutoffs[best_j][best_c])
				self._tfinal = t+1 #you dont always do num_boosting_iterations
				break


			alpha = 0.5*math.log((1.0-epsilon)/epsilon)
			self._alphas.append(0.5*math.log((1.0-epsilon)/epsilon))
			self._hts.append(hjc[best_j][best_c])
			self._j.append(best_j)
			self._c.append(cutoffs[best_j][best_c])
			self._distribution = self.update_distribution(t)
			self._tfinal = t+1
			elapsed = time.time()
			print "\t {} (s)".format(elapsed-start)
		return self

	def predict(self, instance):
		test_feature_vector = instance.get_feature_vector()
		product = 0

		for t in range(self._tfinal):
			alpha_t = self._alphas[t]
			j_t = self._j[t]
			c_t = self._c[t]
			h_t = self._hts[t]
			try: 
				val = test_feature_vector[j_t]
			except KeyError:
				val = 0
			if val > c_t:
				y_hat = h_t[0]
			else:
				y_hat = h_t[1]

			product += alpha_t*y_hat
		if product >= 0:
			output = 1
		else:
			output = 0
		return output

	def compute_c(self, instances):

		dimensions = range(self._max_feature)
		cutoffs = []
		midpoint = lambda vector, index: 0.5*(vector[index+1] + vector[index])
		for j in dimensions:
			jth_column_sorted = sorted(self.get_jth_column(j))
			cutoffs.append([midpoint(jth_column_sorted, i) for i in range(self._n-1)])
		return cutoffs

	def compute_hjc_all(self, cutoffs):
		hjc = []
		j_dimensions = len(cutoffs)
		num_cutoffs = len(cutoffs[0])
		for j in range(j_dimensions):
			hjc_j = []
			for c in range(num_cutoffs):
				hjc_j.append(self.compute_hjc(j, cutoffs[j][c]))
			hjc.append(hjc_j)
		return hjc

	def compute_hjc(self, j, c):
		jth_column = self.get_jth_column(j)

		votes_top = [0, 0]
		votes_bottom = [0, 0]
		for i in range(len(jth_column)):
			label = self._instances[i].get_label()
			val = jth_column[i]
			if val > c:
				votes_top[label] += 1
			else:
				votes_bottom[label] +=1

		if votes_top[1] > votes_top[0]:
			label_top = 1
		else:
			label_top = -1

		if votes_bottom[1] > votes_bottom[0]:
			label_bottom = 1
		else:
			label_bottom = -1
		return ((label_top, label_bottom))

	def compute_errors(self, hjc, cutoffs):
		errors = []
		j_dimensions = len(hjc)
		num_cutoffs = len(cutoffs[0])
		#pick an hjc
		for j in range(j_dimensions):
			error_h = []
			for c in range(num_cutoffs):
				error = 0
				hjc_current = hjc[j][c]
				column_j = self.get_jth_column(j)
		#iterate over all j features of i examples
				for i in range(len(column_j)):
					if column_j[i] > cutoffs[j][c] and hjc[j][c][0] != self._instances[i].get_label():
						error += self._distribution[i]
					elif column_j[i] <= cutoffs[j][c] and hjc[j][c][1] != self._instances[i].get_label():
						error += self._distribution[i]
					else:
						pass
				error_h.append(error)
			errors.append(error_h)
		return errors

	def get_jth_column(self, j):
		arr = []
		for instance in self._instances:
			val = instance.get_value(j)
			arr.append(val)
		return arr


	def min_error(self,errors):
		errors = np.array(errors)
		min_error = min(errors[0])
		min_j = 0
		argmin_c = np.argmin(errors[0])
		for j in range(len(errors)):
			if min(errors[j]) < min_error:
				min_error = min(errors[j])
				min_j = j
				argmin_c = np.argmin(errors[j])
		return ((min_error, min_j, argmin_c))

	def update_distribution(self,t):
		distribution = np.zeros(self._n)
		current_dist = self._distribution
		ht = self._hts[t]
		j = self._j[t]
		c = self._c[t]
		alpha = self._alphas[t]
		for i in range(self._n):
			fv = self._instances[i].get_feature_vector()
			try:
				val = fv[j]
			except KeyError:
				val = 0

			if val > c:
				ht_xi = ht[0]
			else:
				ht_xi = ht[1]

			y_i = self._instances[i].get_label()
			distribution[i] = current_dist[i]*math.exp(-1.0*alpha*y_i*ht_xi)
		Z = sum(distribution)
		distribution = distribution/Z
		return distribution







