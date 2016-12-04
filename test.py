import classify
import subprocess
import os
import time
def run_test(model_file, predictions_file, alg, train=False):
	if train == True:
		start1 = time.time()
		os.system("python classify.py  --mode train --model-file {} --alg {}".format(model_file, alg))
		elapsed1 = time.time() - start1
	else:
		elapsed1 = 0.00
	os.system("python classify.py --mode test --model-file {} --predictions-file {}".format(model_file, predictions_file))
	q = subprocess.check_output("python compute_accuracy.py label_file {}".format(predictions_file),shell=True)
	return (q, elapsed1)

def print_test():
	q = run_test("model_file", "predictions_file", "perceptron", train=True)
	run_time = round(q[1],2)
	str1 = " | "
	str2 = str1.join(("test 1",str(q[0]), str(run_time) + "(s)")).replace("\n", "")
	print str2

	q = run_test("model_file_boosting", "predictions_file", "adaboost", train=False)
	run_time = round(q[1],2)
	str1 = " | "
	str2 = str1.join(("test 2",str(q[0]), str(run_time) + "(s)")).replace("\n", "")
	print str2
print_test()


