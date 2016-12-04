import pandas as pd
from objectTypes import Player, Match

def load_data(number_of_training_matches=1000):
	if number_of_training_matches > 50000:
		print "Data OVERLOADDDDDDDDD"

	path = "dota-2-matches/"

	player_data = pd.read_csv(path+"players.csv", 
		usecols=["match_id", "hero_id", "player_slot", "item_0", "item_1", "item_2", "item_3", "item_4", "item_5"])

	match_data = pd.read_csv(path + "match.csv",
		usecols = ["match_id", "tower_status_radiant", "tower_status_dire", "barracks_status_radiant", "barracks_status_dire", "radiant_win"])

	matches = []
	for i in range(number_of_training_matches): 
		# initialize players in a given match
		players = []
		for j in range(10):
			player = Player()
			player._match_id = player_data["match_id"][(10*i + j)]
			player._hero = player_data["hero_id"][(10*i + j)]
			if player_data["player_slot"][(10*i + j)] < 15:
				player._team = 0
			else:
				player._team = 1
			player._item0 = player_data["item_0"][(10*i + j)]
			player._item1 = player_data["item_1"][(10*i + j)]
			player._item2 = player_data["item_2"][(10*i + j)]
			player._item3 = player_data["item_3"][(10*i + j)]
			player._item4 = player_data["item_4"][(10*i + j)]
			player._item5 = player_data["item_5"][(10*i + j)]
			players.append(player)

		# initialize matches
		match = Match(players)
		match._match_id = match_data["match_id"][i]
		match._tower_status_radiant = "{0:b}".format(match_data["tower_status_radiant"][i]).zfill(16)
		match._tower_status_dire = "{0:b}".format(match_data["tower_status_dire"][i]).zfill(16)
		match._barracks_status_radiant = "{0:b}".format(match_data["barracks_status_radiant"][i]).zfill(8)
		match._barracks_status_dire = "{0:b}".format(match_data["barracks_status_dire"][i]).zfill(8)
		match._label = match_data["radiant_win"][i]

		matches.append(match)

	return matches


def generate_test_data(start=1001, number_of_points=50):

	path = "dota-2-matches/"

	player_data = pd.read_csv(path+"players.csv", 
		usecols=["match_id", "hero_id", "player_slot", "item_0", "item_1", "item_2", "item_3", "item_4", "item_5"])

	match_data = pd.read_csv(path + "match.csv",
		usecols = ["match_id", "tower_status_radiant", "tower_status_dire", "barracks_status_radiant", "barracks_status_dire", "radiant_win"])

	matches = []
	for i in range(number_of_points): 
		# initialize players in a given match
		players = []
		for j in range(10):
			player = Player()
			player._match_id = player_data["match_id"][(start - 1 + 10*i + j)]
			player._hero = player_data["hero_id"][(start - 1 + 10*i + j)]
			if player_data["player_slot"][(start - 1 + 10*i + j)] < 15:
				player._team = 0
			else:
				player._team = 1
			player._item0 = player_data["item_0"][(start - 1 + 10*i + j)]
			player._item1 = player_data["item_1"][(start - 1 + 10*i + j)]
			player._item2 = player_data["item_2"][(start - 1 + 10*i + j)]
			player._item3 = player_data["item_3"][(start - 1 + 10*i + j)]
			player._item4 = player_data["item_4"][(start - 1 + 10*i + j)]
			player._item5 = player_data["item_5"][(start - 1 + 10*i + j)]
			players.append(player)

		# initialize matches
		match = Match(players)
		match._match_id = match_data["match_id"][start - 1 + i]
		match._tower_status_radiant = "{0:b}".format(match_data["tower_status_radiant"][start - 1 + i]).zfill(16)
		match._tower_status_dire = "{0:b}".format(match_data["tower_status_dire"][start - 1 + i]).zfill(16)
		match._barracks_status_radiant = "{0:b}".format(match_data["barracks_status_radiant"][start - 1 + i]).zfill(8)
		match._barracks_status_dire = "{0:b}".format(match_data["barracks_status_dire"][start - 1 + i]).zfill(8)
		match._label = match_data["radiant_win"][start - 1 + i]

		matches.append(match)

	return matches