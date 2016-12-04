import classify
import subprocess
import os
import time
def run_test():
	start1 = time.time()
	os.system("python classify.py  --mode train --model-file model_file")
	elapsed1 = time.time() - start1
	os.system("python classify.py --mode test --model-file model_file --predictions-file predictions_file")
	q = subprocess.check_output("python compute_accuracy.py label_file predictions_file",shell=True)
	return (q, elapsed1)

def print_test():
	q = run_test()
	run_time = round(q[1],2)
	str1 = " | "
	str2 = str1.join(("test 1",str(q[0]), str(run_time) + "(s)")).replace("\n", "")
	print str2
print_test()


