# Main file for classification

from read_data import load_data
from objectTypes import Player, Match


def main():
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

if __name__ == "__main__":
	main()