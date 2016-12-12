import classify
import subprocess
import os
import time
def run_test(name, model_file, predictions_file, alg, train=False):
	if train == True:
		start1 = time.time()
		os.system("python classify.py --data {}.train --mode train --model-file {} --alg {}".format(name, model_file, alg))
		elapsed1 = time.time() - start1
		q = 1
	else:
		elapsed1 = 0.00
	os.system("python classify.py --data {}.dev --mode test --model-file {} --predictions-file {}".format(name, model_file, predictions_file))
	q = subprocess.check_output("python compute_accuracy.py {}.label {}".format(name, predictions_file),shell=True)
	return (q, elapsed1)

def run_SVM(name, train=True):
	if train:
		start = time.time()
		os.system("python SVM.py")
		elapsed = time.time() - start
	else:
		elapsed = 0.0
	q = subprocess.check_output("python compute_accuracy.py dota.2.label SVM.result")
	return (q, elapsed)

def print_test():
	q = run_test("dota.2","model_file_perceptron", "perceptron.result", "perceptron", train=False)
	run_time = round(q[1],2)
	str1 = " | "
	str2 = str1.join(("Perceptron",str(q[0]), str(run_time) + "(s)")).replace("\n", "")
	print str2

	q = run_test("dota.2","model_file_adaboost", "adaboost.result", "adaboostp", train=False)
	run_time = round(q[1],2)
	str1 = " | "
	str2 = str1.join(("AdaBoost P",str(q[0]), str(run_time) + "(s)")).replace("\n", "")
	print str2

	q = run_SVM("dota.2", train=False)
	run_time = round(q[1],2)
	str1 = " | "
	str2 = str1.join(("SVM",str(q[0]), str(run_time) + "(s)")).replace("\n", "")
	print str2

	q = run_test("dota.cluster","model_file_knn", "knn.result", "knn", train=True)
	run_time = round(q[1],2)
	str1 = " | "
	str2 = str1.join(("KNN",str(q[0]), str(run_time) + "(s)")).replace("\n", "")
	print str2
print_test()


