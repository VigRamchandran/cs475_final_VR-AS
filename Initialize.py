# Pre-training commands (Needs to be run only once)
import read_data
reload(read_data)
from read_data import load_data, create_match_file, generate_test_data

def create_label_file(filename, matches):
	with open(filename,'w') as f:
		for match in matches:
			label = None
			if match._label:
				label = '1'
			else:
				label = '0'
			f.write(label + "\n")
	f.close()

# Create training data
matches = load_data(10000)
create_match_file('dota.2.train', matches)

# Create dev/test data
matches = generate_test_data(start=10001,number_of_points=100)
create_match_file('dota.2.dev', matches)
create_label_file('dota.2.label', matches)

