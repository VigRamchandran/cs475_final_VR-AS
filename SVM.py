# SVM from sci-kit learn. Simply to test against our other algorithms

from sklearn import svm
from objectTypes import Data
import numpy as np

def load_instances(filename):
    instances = []
    with open(filename) as reader:
        for line in reader:
            if len(line.strip()) == 0:
                continue
            
            # Divide the line into features and label.
            split_line = line.split(" ")
            label_string = split_line[0]

            int_label = -1
            try:
                int_label = int(label_string)
            except ValueError:
                raise ValueError("Unable to convert " + label_string + " to integer.")

            data_point = Data()
            data_point.set_label(int_label)
            
            for item in split_line[1:]:
                try:
                    index = int(item.split(":")[0])
                except ValueError:
                    continue
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    continue
                
                if value != 0.0:
                    data_point.add_feature(index, value)

            
            instances.append(data_point)

    return instances

instances = load_instances("dota.2.train")

def instances_to_array(instances):
    curr_max = 1
    for instance in instances:
        if instance.get_max_feature() > curr_max:
            curr_max = instance.get_max_feature()

    examples = []
    labels = []
    for i in range(len(instances)):
        fv_arr = np.zeros(curr_max)
        fv = instances[i].get_feature_vector()
        for index in fv:
            fv_arr[index-1] = fv[index]
        examples.append(fv_arr)
        labels.append(instances[i].get_label())

    return examples, labels

X, Y = instances_to_array(instances)
clf = svm.SVC()
clf.fit(X, Y)

testing = load_instances("dota.2.dev")
X2, Y2 = instances_to_array(testing)
with open("SVM.result", 'w') as f:
	for i in X2:
	    f.write(str(clf.predict([i])[0]) + "\n")
f.close()

	