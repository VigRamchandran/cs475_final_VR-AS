# Contains class definitions for node objects used

import numpy as np
import pandas as pd
import sys
import math


class Player:
    def __init__(self):
        self._match_id = None
        self._hero = None
        self._team = None
        self._item0 = None
        self._item1 = None
        self._item2 = None
        self._item3 = None
        self._item4 = None
        self._item5 = None
        self._purchase = None


class Match:
    def __init__(self, players):
        self._players = players
        self._match_id = 0
        self._tower_status_radiant = 0  # will be a 16-bit array
        self._tower_status_dire = 0
        self._barracks_status_radiant = 0  # will be an 8-bit array
        self._barracks_status_dire = 0
        self._label = 0
        self._objectives = None
        self.num_heros = 113
        self.num_items = 265 + 1  # Extra 1 for zero pad, zero refers to no item
        self.max_game_length = 7200  # Assume max game len 2 hrs
        self.bin_size = 300  # Bin size of 5m
        self.items = (46, 42)  # Track purchases of teleport scrolls, observer wards by player (may be better by team)
        # self._obj_feat = None
        # self._firstblood = 0  # 0 for radiant, 1 for dire

    def get_feature_vector(self, player_feat=1, purchase_feat=1, obj_time=1, obj_end=0):
        feature_vector = []


        for player in self._players:
            if player_feat:
                player_features = self.transform_player(player)
                feature_vector.extend(player_features)

            if purchase_feat:
                purchases = player._purchase
                purchase_features = self.transform_purchases(purchases, self.items,
                                                             max_time=self.max_game_length, bin_size=self.bin_size)
                feature_vector.extend(purchase_features)


        if obj_time:
            objective_features, fb = self.transform_objectives(max_time=900, bin_size=self.bin_size)
            feature_vector.extend([fb])
            feature_vector.extend(objective_features)

        if obj_end:
            tower_radiant = self.bit2arr(self._tower_status_radiant)
            tower_dire = self.bit2arr(self._tower_status_dire)
            barracks_radiant = self.bit2arr(self._barracks_status_radiant)
            barracks_dire = self.bit2arr(self._barracks_status_dire)

            feature_vector.extend(tower_radiant)
            feature_vector.extend(tower_dire)
            feature_vector.extend(barracks_radiant)
            feature_vector.extend(barracks_dire)

        return feature_vector

    def transform_player(self, player):
        features = []
        hero = [0] * self.num_heros
        hero[player._hero] = 1
        features.extend(hero)

        items = [0] * self.num_items

        items[player._item0] += 1
        items[player._item1] += 1
        items[player._item2] += 1
        items[player._item3] += 1
        items[player._item4] += 1
        items[player._item5] += 1
        features.extend(items)

        return features


    def transform_objectives(self, max_time=7200.0, bin_size=300.0):

        buckets = int(math.ceil(max_time/bin_size))

        features = [0]*6*(buckets)  # [0, 0, 0, 0, 0, 0] -> Rad Tower/Barracks/Roshan, Dire Tower/Barracks/Roshan
        # Cap it out at 2 hr games. Past that assume no predictive capabilities because its based on so much.
        fb = None

        for index, row in self._objectives.iterrows():
            if row['subtype'] == 'CHAT_MESSAGE_TOWER_KILL':
                time = int(row['time'])
                i,j = divmod(time, bin_size)
                team = int(row['team'])-2
                ind = int((i*6)+(team*3))
                if ind < len(features):
                    features[ind] += 1
                else:
                    pass
            if row['subtype'] == 'CHAT_MESSAGE_BARRACKS_KILL':
                time = int(row['time'])
                i, j = divmod(time, bin_size)
                if row['key'] <= 32:  # Powers of Two, Dire->Radiant, Bottom -> Top, Melee-> Ranged.
                    team = 1
                else:
                    team = 0
                ind = int((i*6) + (team * 3) + 1)
                if ind < len(features):
                    features[ind] += 1
                else:
                    pass
            if row['subtype'] == 'CHAT_MESSAGE_ROSHAN_KILL':
                time = int(row['time'])
                i,j = divmod(time, bin_size)
                team = int(row['team'])-2
                ind = int((i*6)+(team*3) + 2)
                if ind < len(features):
                    features[ind] += 1
                else:
                    pass
            if row['subtype'] == 'CHAT_MESSAGE_FIRSTBLOOD':
                if row['player1'] < 5:
                    fb = 1
                else:
                    fb = 0

        # self._obj_feat = features
        # self._firstblood = fb
        return features, fb

    def transform_purchases(self, purchase, items, max_time=7200.0, bin_size=300):
        buckets = int(math.ceil(max_time / bin_size))
        pdarray = purchase.as_matrix()

        features = [0] * len(items) * (buckets+1)  # buckets + 1 since we are including ones bought before game starts

        for i in range(-1, buckets):
            pdbucket = pdarray[(pdarray[:,1] > i*bin_size) & (pdarray[:,1] < (i+1)*bin_size), :2]
            for j in range(len(items)):
                buys = np.sum(pdbucket[:, 0] == items[j])
                features[(i+1)*len(items) + j] = buys

        return features

    def get_label(self):
        return int(self._label)

    def bit2arr(self, bit_str):  # converts a bit string to an array.
        arr = (",").join(bit_str).split(",")
        for i in range(len(arr)):
            arr[i] = int(arr[i])

        return arr

class Data:
    def __init__(self):
        self._feature_vector = {}
        self._label = None
    def add_feature(self, index, value):
        self._feature_vector[index] = value
    def set_label(self, label):
        self._label = int(label)
    def get_label(self):
        return self._label
    def get_feature_vector(self):
        return self._feature_vector
