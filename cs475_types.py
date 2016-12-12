from abc import ABCMeta, abstractmethod
import numpy as np
import time
import math
import sys
from scipy.stats import norm

# abstract base class for defining labels
class Label:
    __metaclass__ = ABCMeta

    @abstractmethod
    def __str__(self): pass

       
class ClassificationLabel(Label):
    def __init__(self, label):
        self.label = label
        return
        
    def __str__(self):
        return str(self.label)
        pass

    def val(self):
        if int(self.label) == 0:
            return -1
        else:
            return self.label


class FeatureVector:
    def __init__(self):
        self.all_values = {}
        pass

    def __str__(self):
        return str(self.all_values)
        
    def add(self, index, value):
        self.all_values[index] = value
        pass
        
    def get(self, index):
        if index in self.all_values:
            return self.all_values[index]
        else:
            return 0.0

    def all_vals(self):
        return self.all_values

    def get_largest_index(self):
        return max(k for k, v in self.all_values.items())
        

class Instance:
    def __init__(self, feature_vector, label):
        self._feature_vector = feature_vector
        self._label = label
        self.max_index = 0

    def get_feature_vector(self):
        return self._feature_vector

    def get_label(self):
        return self._label

    def get_max_index(self):
        return self._feature_vector.get_largest_index()


# abstract base class for defining predictors
class Predictor:
    __metaclass__ = ABCMeta

    @abstractmethod
    def train(self, instances): pass

    @abstractmethod
    def predict(self, instance): pass


class Perceptron(Predictor):
    def __init__(self, learn_rate, iterations):
        self.learn_rate = learn_rate
        self.iter = iterations
        self.w = {}
        self.max_index = 0
        return

    def train(self, instances):
        if self.max_index == 0:
            self.get_max_index(instances)

        # all_losses = []

        for i in range(self.iter):
            w_use = self.train_one_round(instances, self.w)
            # all_losses.append(total_loss)
            self.w = w_use

        return

    def train_one_round(self, instances, w):
        w_use = w

        for inst in instances:
            dot = self.dot_prod(inst)
            # xi = self.create_vector(inst)
            # dot = sum(x[0] * x[1] for x in zip(xi, w_use))
            if dot >= 0:
                yhat = 1
            else:
                yhat = -1

            y = inst.get_label().val()
            # loss = max(0, -1 * y * yhat)
            # total_loss += loss

            if y != yhat:
                self.w = self.w_update(inst, self.w, y, yhat)

        return w_use

    def dot_prod(self, inst):
        val_dict = inst.get_feature_vector().all_vals()
        tot_sum = 0
        for key in val_dict:
            if key in self.w:
                tot_sum += val_dict[key] * self.w[key]
            else:
                self.w[key] = 0

        return tot_sum

    def w_update(self, inst, w, y, yhat):
        val_dict = inst.get_feature_vector().all_vals()
        new_w = w
        for key in val_dict:
            new_w[key] += self.learn_rate*(y-yhat)*val_dict[key]

        return new_w

    def predict(self, instance):
        if self.dot_prod(instance) >= 0:
            return 1
        else:
            return 0

    def get_max_index(self, instances):
        for i in instances:
            if i.get_max_index() > self.max_index:
                self.max_index = i.get_max_index()
        return


class MC_Perceptron(Predictor):
    def __init__(self, iterations):
        self.w = None
        self.N = 0  # Num of instances
        self.K = 0  # Num of classes. If 0 and 1, will just return 1. May need an i+1 for iterations.
        self.M = 0  # max_index
        self.inst = None
        self.iter = iterations
        self.labels = None
        return

    def fill_parameters(self, instances):
        for i in instances:
            if i.get_max_index() > self.M:
                self.M = i.get_max_index()

            if i.get_label().val() > self.K:
                self.K = i.get_label().val()
        return

    def train(self, instances):
        self.N = len(instances)

        self.fill_parameters(instances)

        self.w = np.zeros((self.M, self.K))

        self.inst = np.zeros((self.N, self.M))
        self.labels = np.zeros(self.N)
        for i in range(self.N):
            instance = instances[i]
            vals = instance.get_feature_vector().all_vals()
            array = self.dict_to_array(vals)
            self.inst[i, :] = array[1:]
            self.labels[i] = instance.get_label().val()
            if self.labels[i] == -1:
                self.labels[i] = 0  # We don't want neg 1 labels here

        for i in range(self.iter):

            for j in range(self.N):
                prod = self.inst.dot(self.w)
                guess = np.argmax(prod[j, :])
                value = int(self.labels[j]) - 1  # -1 because our index goes from 1-1 through K-1

                if guess == value:  # If we guessed it right
                    # print 'Train was right'
                    pass
                else:
                    self.w[:, guess] = self.w[:, guess] - self.inst[j, :]
                    self.w[:, value] = self.w[:, value] + self.inst[j, :]

    def dict_to_array(self, vals):
        temp = np.array([0.0] * (self.M + 1))  # index 0 is never used, no feature is index 0

        for key in vals:
            try:
                temp[key] = vals[key]
            except IndexError:  # Ignore when more features in test than train
                pass

        return temp

    def predict(self, instance):
        array = self.dict_to_array(instance.get_feature_vector().all_vals())
        guesses = array[1:].dot(self.w)
        best = np.argmax(guesses) + 1
        return best


class Averaged_Perceptron(Perceptron):
    def __init__(self, learn_rate, iterations):
        super(Averaged_Perceptron, self).__init__(learn_rate, iterations)
        self.w_total = {}
        return

    def train(self, instances):
        if self.max_index == 0:
            self.get_max_index(instances)

        # all_losses = []

        for i in range(self.iter):
            w_use = self.train_one_round(instances, self.w)
            # all_losses.append(total_loss)
            self.w = w_use

        return

    def train_one_round(self, instances, w):
        w_use = w

        for inst in instances:

            dot = self.dot_prod(inst)
            # xi = self.create_vector(inst)
            # dot = sum(x[0] * x[1] for x in zip(xi, w_use))
            if dot >= 0:
                yhat = 1
            else:
                yhat = -1

            y = inst.get_label().val()
            # loss = max(0, -1 * y * yhat)
            # total_loss += loss

            if y != yhat:
                self.w = self.w_update(inst, self.w, y, yhat)

            for k, v in self.w.iteritems():
                if k in self.w_total:
                    self.w_total[k] += v
                else:
                    self.w_total[k] = v

            # for key in self.w:
            #     if key in self.w_total:
            #         self.w_total[key] += self.w[key]
            #     else:
            #         self.w_total[key] = self.w[key]
        return w_use

    def predict(self, instance):
        val_dict = instance.get_feature_vector().all_vals()
        tot_sum = 0
        for key in val_dict:
            if key in self.w_total:
                tot_sum += val_dict[key] * self.w_total[key]
            else:
                tot_sum += 0  # If you've never trained it, should stay at 0

        if tot_sum >= 0:
            return 1
        else:
            return 0


class Margin_Perceptron(Perceptron):
    def __init__(self, learn_rate, iterations):
        super(Margin_Perceptron, self).__init__(learn_rate, iterations)
        return

    def train_one_round(self, instances, w):
        w_use = w

        for inst in instances:
            dot = self.dot_prod(inst)

            y = inst.get_label().val()

            if y * dot < 1:
                self.w = self.w_update(inst, self.w, y, 1)

        return w_use

    def w_update(self, inst, w, y, yhat):
        val_dict = inst.get_feature_vector().all_vals()
        new_w = w
        for key in val_dict:
            new_w[key] += self.learn_rate * y * val_dict[key]

        return new_w


class Pegasos(Perceptron):
    def __init__(self, learn_rate, iterations, lamb):
        super(Pegasos, self).__init__(learn_rate, iterations)
        self.t = 1
        self.lamb = lamb
        return

    def train_one_round(self, instances, w):
        w_use = w

        for inst in instances:
            dot = self.dot_prod(inst)

            y = inst.get_label().val()  # Returns 1 if 1, -1 if 0

            self.w = self.w_update(inst, self.w, y, dot)

            self.t += 1.0
        return w_use

    def loss(self, dot, y):
        a = dot * y
        if a < 1:
            return 1
        else:
            return 0

    def w_update(self, inst, w, y, dot):
        val_dict = inst.get_feature_vector().all_vals()
        new_w = w

        for key in new_w:
            update = (1.0 - (1.0/self.t)) * new_w[key]
            if key in val_dict:
                update += (1.0 / (self.lamb * self.t)) * self.loss(dot, y) * y * val_dict[key]
            new_w[key] = update

        return new_w


class KNN(Predictor):

    def __init__(self, k):
        self.k = k
        self.max_index = 0
        self.instances = None
        self.tot_dist_calc = 0
        self.arrays = []

    def train(self, instances):
        self.instances = instances

        for inst in instances:
            inst_max = inst.get_max_index()
            if inst_max > self.max_index:
                self.max_index = inst_max

        for inst in instances:
            self.arrays.append(self.dict_to_array(inst.get_feature_vector().all_vals()))
        return

    def dict_to_array(self, vals):
        temp = np.array([0.0] * (self.max_index + 1))  # index 0 is never used, no feature is index 0

        for key in vals:
            try:
                temp[key] = vals[key]
            except IndexError:  # Ignore when more features in test than train
                pass

        return temp

    def predict(self, instance):
        comparisons = []

        test_array = self.dict_to_array(instance.get_feature_vector().all_vals())
        for i in range(len(self.instances)-1):

            dist = np.linalg.norm(test_array-self.arrays[i])
            # dist = np.sqrt(np.sum((test_array - self.arrays[i]) ** 2))  # Distance calc
            comparisons.append([self.instances[i].get_label().val(), dist])  # Append [label, distance]

        comparisons = sorted(comparisons, key=lambda key: (key[1], key[0]))

        comparisons_trimmed = comparisons[0:self.k]
        # print comparisons_trimmed

        guess = self.classification(comparisons_trimmed)
        return guess

    def classification(self, top_k):
        toppers = {}
        guesses = []
        max_votes = 0
        for a in top_k:
            try:
                toppers[a[0]] += 1  # The guess is a[0], it gets one more vote
            except KeyError:
                toppers[a[0]] = 1

        for key in toppers:  # Find the highest number of vites
            if toppers[key] > max_votes:
                max_votes = toppers[key]

        for k, v in toppers.items():
            if v == max_votes:
                guesses.append(k)

        guess = min(guesses)

        if guess == -1:
            guess = 0

        return guess


class dKNN(KNN):
    def __init__(self, k):
        super(dKNN, self).__init__(k)

    def classification(self, top_k):
        toppers = {}
        guesses = []
        max_score = 0
        for a in top_k:
            try:
                toppers[a[0]] += 1.0 / (1.0 + a[1])  # The guess is a[0], it gets one more vote weighted on distance
            except KeyError:
                toppers[a[0]] = 1.0 / (1.0 + a[1])

        for key in toppers:  # Find the highest number of vites
            if toppers[key] > max_score:
                max_score = toppers[key]

        for k, v in toppers.items():
            if v == max_score:
                guesses.append(k)

        guess = min(guesses)

        if guess == -1:
            guess = 0

        return guess


class AdaBoost(Predictor):
    def __init__(self, t):
        self.T = t  # Number of boosting iterations
        self.D = None
        self.instances = None
        self.max_index = 0
        self.N = 0
        self.all_train = None
        self.c = None
        self.labels = None
        self.pred_alphas = np.zeros(self.T, dtype=np.float)
        self.pred_hs = np.zeros(self.T, dtype=np.float)
        self.pred_c = np.zeros(self.T, dtype=np.float)
        self.pred_classifications = np.zeros((self.T, 2), dtype=np.float)

    def train(self, instances):
        self.instances = instances
        self.N = len(instances)

        for inst in instances:
            inst_max = inst.get_max_index()
            if inst_max > self.max_index:
                self.max_index = inst_max

        self.D = np.ones(len(instances), dtype=np.float) * (1.0 / len(instances))  # Initialize D1
        self.all_train = np.zeros((len(instances), self.max_index + 1))  # Store all training examples in np array
        self.c = np.zeros(self.max_index + 1)  # vector of cutoffs
        self.labels = np.zeros(len(instances))

        for i in range(len(instances)):  # Get a list of all labels, Populate example array
            self.labels[i] = instances[i].get_label().val()
            self.all_train[i, :] = self.dict_to_array(instances[i].get_feature_vector().all_vals())

        # print 'alltrain before trim: ', self.all_train

        self.all_train = np.delete(self.all_train, 0, 1)  # Because features start at 1, but our np array starts at 0

        print 'Caching c and creating all hs'
        c_final, classification = self.get_classifications()  # Get the cutoff rules for each c
        # errors_temp = self.get_errors(c_final, classification)
        errors = 0
        for i in range(self.T):
            print 'Boosting Iteration ', i+1
            errors = self.get_errors(c_final, classification)
            a, b = np.unravel_index(errors.argmin(), errors.shape)
            c_to_use = c_final[a, b]
            eps = errors[a, b]
            h_t = b  # Feature vector to use
            temp_classify = classification[a, b]

            if eps < .000001:
                print 'Tiny eps'
                if i == 0:  # Need to account for edge case where our first hypothesis is perfect
                    self.pred_alphas[i] = 1
                    self.pred_hs[i] = h_t
                    self.pred_classifications[i] = temp_classify
                    self.pred_c[i] = c_to_use
                    # print 'alphas: ', self.pred_alphas
                    # print 'hs: ', self.pred_hs
                    # print 'pred_classifications: ', self.pred_classifications
                    # print 'c: ', self.pred_c
                    break
                break

            self.pred_alphas[i] = .5 * math.log((1 - eps) / eps)
            self.D = self.update_ds(self.pred_alphas[i], h_t, c_to_use, temp_classify)
            self.pred_hs[i] = h_t
            self.pred_c[i] = c_to_use
            self.pred_classifications[i] = temp_classify

        return errors, c_final, classification

    def predict(self, instance):
        arr = self.dict_to_array(instance.get_feature_vector().all_vals())

        arr = np.delete(arr, 0)  # No feature number 0
        tot = 0
        for i in range(self.T):
            alpha_t = self.pred_alphas[i]
            classify_t = self.pred_classifications[i]
            c_t = self.pred_c[i]
            h_t = int(self.pred_hs[i])

            if arr[h_t] > c_t:
                yhat = classify_t[0]
            else:
                yhat = classify_t[1]

            tot += alpha_t * yhat
        if tot >= 0:
            return 1
        else:
            return 0

    def h(self, i, j, c, classification):
        if self.all_train[i, j] > c:
            return classification[0]
        elif self.all_train[i, j] <= c:
            return classification[1]

    def get_errors(self, c_final, classification):
        errors = np.zeros((self.N-1, self.max_index), dtype=np.float)
        for i in range(self.max_index):
            for j in range(self.N-1):
                test_c = c_final[j, i]
                error = 0
                for k in range(self.N):
                    if self.h(k, i, test_c, classification[j, i]) != self.labels[k]:
                        error += float(self.D[k])
                    else:
                        pass
                errors[j, i] = error
        return errors

    def update_ds(self, alpha, h_t, c, temp_classify):
        new_ds = np.zeros(len(self.instances), dtype=np.float)

        for ex in range(self.N):
            new_d = self.D[ex]
            coeff = math.exp(-1.0 * alpha * self.labels[ex] * self.h(ex, h_t, c, temp_classify))
            new_d *= coeff
            new_ds[ex] = new_d

        norm_factor = sum(new_ds)
        new_ds /= norm_factor

        return new_ds

    def get_classifications(self):
        c_final = np.zeros((self.N-1, self.max_index), dtype=np.float)
        classification = np.zeros((self.N-1, self.max_index, 2), dtype=np.float)

        for i in range(self.max_index):  # Across each of the feature vectors
            sorted_features = np.array(sorted(self.all_train[:, i]))

            # if i == 0:  # c matches up to order
            #     print 'sorted features for first: ', sorted_features

            test_c = (sorted_features[1:] + sorted_features[:-1]) / 2.0
            # print 'test_c: ', test_c  # test_c calculated correctly

            c_final[:, i] = test_c
            # print 'cfinal: ', c_final  # c_final is built correctly

            for k in range(self.N-1):
                classification[k, i] = self.get_c_direction(test_c[k], i)

        # print 'Is classification ever not assigned? ', np.where(classification == 0)  # This is fine

        return c_final, classification

    def get_c_direction(self, c, feat):  # Checked, should be correct
        y_above_one = 0
        y_below_one = 0
        y_above_negone = 0
        y_below_negone = 0
        direction = np.zeros(2)
        for k in range(self.N):
            if self.all_train[k, feat] > c:
                if self.labels[k] == 1:
                    y_above_one += 1
                else:
                    y_above_negone += 1
            elif self.all_train[k, feat] <= c:
                if self.labels[k] == 1:
                    y_below_one += 1
                else:
                    y_below_negone += 1

        # print 'c: ', c
        # print 'data: ', self.all_train[:,feat]
        # print 'labels: ', self.labels
        # print 'y above 1, y_above -1, y_below, y_below -1: ', y_above_one, y_above_negone, y_below_one, y_below_negone

        if y_above_one >= y_above_negone:
            direction[0] = 1
        else:
            direction[0] = -1

        if y_below_one >= y_below_negone:
            direction[1] = 1
        else:
            direction[1] = -1
        return direction  # 2-length array that is [prediction if >c, prediction if <= c]

    def dict_to_array(self, vals):
        temp = np.array([0.0] * (self.max_index + 1))  # index 0 is never used, no feature is index 0

        for key in vals:
            try:
                temp[key] = vals[key]
            except IndexError:  # Ignore when more features in test than train
                pass

        return temp


class Lambda_Means(Predictor):
    def __init__(self, lamb, iter):
        self.r = None
        self.means = None
        self.k = 1  # True value of number of clusters
        self.arrays = []
        self.lamb = lamb
        self.max_index = 0
        self.n = None
        self.iter = iter

    def train(self, instances):

        for inst in instances:
            inst_max = inst.get_max_index()
            if inst_max > self.max_index:
                self.max_index = inst_max

        for inst in instances:
            self.arrays.append(self.dict_to_array(inst.get_feature_vector().all_vals()))

        self.n = len(self.arrays)  # n is the total number of instances

        self.arrays = np.array(self.arrays)
        self.arrays = np.delete(self.arrays, 0, 1)
        self.r = np.ones((1, self.n))  # Create the first row of the n vector
        self.means = self.r.dot(self.arrays) / np.sum(self.r, 1)

        if self.lamb == 0.0:  # If the default value of 0.0 is passed to the constructor
            init_lambda = 0
            for i in range(self.n):
                init_lambda += np.sum(np.square(self.means-self.arrays[i]))
                # init_lambda += self.distance(self.means, self.arrays[i])
            self.lamb = init_lambda / float(self.n)

        for i in range(self.iter):
            self.EM()

        return

    def EM(self):
        old_r = self.r
        self.r.fill(0)  # Set it all to zero
        for i in range(self.n):
            test_instance = self.arrays[i]
            dist = np.zeros(self.k)

            for j in range(self.k):
                test_mean = self.means[j]
                dist[j] = self.distance(test_instance, test_mean)**2

            if np.all(dist > self.lamb):  # If every distance to every mean > lambda
                self.r = np.vstack((self.r, np.zeros((1, self.n))))  # Add a new row to r
                self.means = np.vstack((self.means, test_instance))  # add a new mean
                self.r[self.k, i] = 1  # Indexing to k effectively goes to the new row
                self.k += 1
                test = np.all(dist != self.lamb)
                if ~test:
                    print 'Some distance is same as lamb: ', dist
            else:
                c_cluster = np.argmin(dist)
                self.r[c_cluster, i] = 1
        new_r = self.r

        tot = np.sum(self.r, 1)
        tot = tot[:, None]

        emp_clusters = np.where(tot == 0)
        tot[emp_clusters] = .0001  # Make the value slightly more than 0 to avoid DivByZero errors

        self.means = self.r.dot(self.arrays) / tot  # Automatically sets empty cluster means to 0

        return old_r, new_r

    def dict_to_array(self, vals):
        temp = np.array([0.0] * (self.max_index + 1))  # index 0 is never used, no feature is index 0

        for key in vals:
            try:
                temp[key] = vals[key]
            except IndexError:  # Ignore when more features in test than train
                pass

        return temp

    def predict(self, instance):
        arr = self.dict_to_array(instance.get_feature_vector().all_vals())
        arr = np.delete(arr, 0)
        distances = np.zeros(len(self.means))
        for i in range(len(self.means)):
            distances[i] = self.distance(arr, self.means[i])**2

        return np.argmin(distances)

    def distance(self, arr1, arr2):
        dist = np.linalg.norm(arr1-arr2)
        return dist


class NBClustering(Predictor):
    def __init__(self, k, iterations):
        self.K = k  # Number of folds / clusters
        self.iter = iterations  # Number of iterations
        self.arrays = []
        self.n = None
        self.max_index = 0
        self.r = None
        self.means = None
        self.sigmas = None
        self.phat = None
        self.Sj = None

    def train(self, instances):

        for inst in instances:
            inst_max = inst.get_max_index()
            if inst_max > self.max_index:
                self.max_index = inst_max

        self.n = len(instances)
        self.r = np.zeros(self.n)
        self.means = np.zeros((self.K, self.max_index))
        self.arrays = np.zeros((self.n, self.max_index + 1))  # Data starts at index 1
        self.sigmas = np.zeros((self.K, self.max_index))
        self.phat = np.zeros(self.K)

        for i in range(len(instances)):
            self.arrays[i] = (self.dict_to_array(instances[i].get_feature_vector().all_vals()))
            self.r[i] = (i+1) % self.K  # Initial fold assignments

        self.arrays = np.delete(self.arrays, 0, 1)

        self.Sj = .01 * np.var(self.arrays, 0, ddof=0)
        self.Sj[self.Sj == 0] = 1

        for i in range(self.K):
            cluster = i
            temp = self.arrays[self.r == cluster]
            self.means[cluster] = np.mean(temp, 0)
            if len(temp) > 2:
                a = np.var(temp, 0, ddof=1)  # Actual unbiased variance
                self.sigmas[cluster] = np.maximum(a, self.Sj)  # Element by element maximum
            else:
                self.sigmas[cluster] = self.Sj
            self.phat[i] = (float(len(temp)) + 1) / (self.n + self.K)

        for i in range(self.iter):
            # print 'iteration: ', i
            for j in range(self.n):
                arr = self.arrays[j]
                new_cluster, a = self.expectation(arr)
                self.r[j] = new_cluster

            self.maximization()

    def expectation(self, arr):
        yhat = np.zeros(self.K)

        py_means = self.means.tolist()
        py_sigma = self.sigmas.tolist()
        for j in range(self.K):  # Clusters
            loglikli = 0
            for i in range(self.max_index):
                try:
                    if py_sigma[j][i] == 0:
                        print 'Sigma 0'
                    const = -math.log(math.sqrt(2 * py_sigma[j][i] * math.pi))
                    val = -((arr[i] - py_means[j][i]) ** 2) / (2 * py_sigma[j][i])
                    loglikli += (const + val)
                    # loglikli += math.log(norm.pdf(arr[i], loc=py_means[j][i], scale=math.sqrt(py_sigma[j][i])))
                except ValueError:
                    loglikli = float('-inf')

            yhat[j] = math.log(self.phat[j]) + loglikli

        new_cluster = np.argmax(yhat)
        return new_cluster, yhat

    def maximization(self):
        for i in range(self.K):
            self.phat[i] = float(sum(self.r == i) + 1) / (self.n + self.K)

            temp = self.arrays[self.r == i]

            if len(temp) == 0:
                self.means[i] = np.zeros((1, self.max_index))
                self.sigmas[i] = self.Sj
            else:
                self.means[i] = np.mean(temp, 0)

            if len(temp) > 2:
                a = np.var(temp, 0, ddof=1)  # Actual unbiased variance
                self.sigmas[i] = np.maximum(a, self.Sj)  # Element by element maximum
            else:
                self.sigmas[i] = self.Sj

        return

    def dict_to_array(self, vals):
        temp = np.array([0.0] * (self.max_index + 1))  # index 0 is never used, no feature is index 0

        for key in vals:
            try:
                temp[key] = vals[key]
            except IndexError:  # Ignore when more features in test than train
                pass

        return temp

    def predict(self, instance):
        arr = self.dict_to_array(instance.get_feature_vector().all_vals())
        arr = np.delete(arr, 0)

        new_cluster, yhat = self.expectation(arr)

        return new_cluster