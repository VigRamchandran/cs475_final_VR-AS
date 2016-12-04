# Main file for classification

from read_data import load_data, generate_test_data
from objectTypes import Player, Match
from Perceptron import Perceptron
from AdaBoost import AdaBoost

import os
import argparse
import sys
import pickle

def load_instances():
    matches = load_data(1000)
    training_examples = []
    training_labels = []
    for match in matches:
        fv = match.get_feature_vector()
        training_examples.append(fv)
        label = match.get_label()
        if label == 0:
            label = -1
        else:
            label = 1
        training_labels.append(label)
    return (training_examples, training_labels)

def get_args():
    parser = argparse.ArgumentParser(description="This is the main test harness for your algorithms.")
    parser.add_argument("--mode", type=str, required=True, choices=["train", "test"],
                        help="Operating mode: train or test.")
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
            with open("label_file",'w') as writer2:
                for instance in instances:
                    testing_example = instance.get_feature_vector()
                    correct_label = instance.get_label()
                    label = predictor.predict(testing_example)
            
                    writer.write(str(label))
                    writer.write('\n')
                    writer2.write(str(correct_label))
                    writer2.write('\n')
    except IOError:
        raise Exception("Exception while opening/writing file for writing predicted labels: " + predictions_file)


def main():
    args = get_args()

    if args.mode.lower() == "train":
        # Load the training data.
        instances = load_instances()

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
        instances = generate_test_data(1001, 500)

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

