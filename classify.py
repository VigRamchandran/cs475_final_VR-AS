# Main file for classification

from read_data import load_data, generate_test_data
from objectTypes import Player, Match, Data
from Perceptron import Perceptron
from AdaBoost import AdaBoost

import os
import argparse
import sys
import pickle

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
                    raise ValueError("Unable to convert index " + item.split(":")[0] + " to integer.")
                try:
                    value = float(item.split(":")[1])
                except ValueError:
                    raise ValueError("Unable to convert value " + item.split(":")[1] + " to float.")
                
                if value != 0.0:
                    data_point.add_feature(index, value)

            
            instances.append(data_point)

    return instances

def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
    parser.add_argument("--data",type=str,required = True, help="Name of data file")
    parser.add_argument("--label_file", type=str, help="Name of file containing true labels for testing")
    parser.add_argument("--model-file", type=str, required=True,
                        help="The name of the model file to create/load.")
    parser.add_argument("--predictions-file", type=str, help="The predictions file to create.")
    parser.add_argument("--alg", type=str, help="desired algorithm.")
    args = parser.parse_args()

    return args


def train(instances, alg):
    if alg == "adaboost":
        p = AdaBoost(2)
    elif alg == "perceptron":
        p = Perceptron()
    training_examples = instances[0]
    training_labels = instances[1]
    p = p.train(training_examples, training_labels)
    return p


def write_predictions(predictor, instances, predictions_file):
    try:
        with open(predictions_file, 'w') as writer:
            for instance in instances:
                label = predictor.predict(instance)
                writer.write(str(label))
                writer.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
    args = get_args()
    case = 2
    if args.mode.lower() == "train":
        # Load the training data.
        instances = load_instances(args.data)

        # Train the model.
        predictor = train(instances,args.alg)
        try:
            with open(args.model_file, 'wb') as writer:
                pickle.dump(predictor, writer)
        except IOError:
            raise Exception("Exception while writing to the model file.")        
        except pickle.PickleError:
            raise Exception("Exception while dumping pickle.")
            
    elif args.mode.lower() == "test":
        # Load the test data. Only want the feature vector this time.
        instances = load_instances(args.data)

        predictor = None
        # Load the model.
        try:
            with open(args.model_file, 'rb') as reader:
                predictor = pickle.load(reader)
        except IOError:
            raise Exception("Exception while reading the model file.")
        except pickle.PickleError:
            raise Exception("Exception while loading pickle.")
            
        write_predictions(predictor, instances, args.predictions_file)
    else:
        raise Exception("Unrecognized mode.")

if __name__ == "__main__":
    main()

