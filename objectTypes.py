# Contains class definitions for node objects used

import numpy as np

class Player:
	def __init__(self):
		self._match_id = None
		self._hero = None
		self._team = None
		self._item0= None
		self._item1 = None
		self._item2 = None
		self._item3 = None
		self._item4 = None
		self._item5 = None


class Match:
	def __init__(self, players):
		self._players = players
		self._match_id = 0
		self._tower_status_radiant = 0
		self._tower_status_dire = 0
		self._barracks_status_radiant = 0
		self._barracks_status_dire = 0
		self._label = 0

	def get_feature_vector(self):
		return 1

