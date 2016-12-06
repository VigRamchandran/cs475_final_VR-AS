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
        self._obj_feat = None
        self._firstblood = 0  # 0 for radiant, 1 for dire

    def get_feature_vector(self, case):
        feature_vector = []
        if case == 1:

            for player in self._players:
                feature_vector.append(player._hero)
                feature_vector.append(player._item0)
                feature_vector.append(player._item1)
                feature_vector.append(player._item2)
                feature_vector.append(player._item3)
                feature_vector.append(player._item4)
                feature_vector.append(player._item5)

            tower_radiant = self.bit2arr(self._tower_status_radiant)
            tower_dire = self.bit2arr(self._tower_status_dire)
            barracks_radiant = self.bit2arr(self._barracks_status_radiant)
            barracks_dire = self.bit2arr(self._barracks_status_dire)

            for num in tower_radiant:
                feature_vector.append(num)
            for num in tower_dire:
                feature_vector.append(num)
            for num in barracks_dire:
                feature_vector.append(num)
            for num in barracks_radiant:
                feature_vector.append(num)
        elif case == 2:
            for player in self._players:
                feature_vector.append(player._hero)
                feature_vector.append(player._item0)
                feature_vector.append(player._item1)
                feature_vector.append(player._item2)
                feature_vector.append(player._item3)
                feature_vector.append(player._item4)
                feature_vector.append(player._item5)
        else:
            print "Invalid case"

        return feature_vector

    def transform_objectives(self, bin=300):

        buckets = int(math.ceil(7200.0/300))

        features = [0]*6*(buckets)  # [0, 0, 0, 0, 0, 0] -> Rad Tower/Barracks/Roshan, Dire Tower/Barracks/Roshan
        # Cap it out at 2 hr games. Past that assume no predictive capabilities because its based on so much.
        fb = None

        for index, row in self._objectives.iterrows():
            if row['subtype'] == 'CHAT_MESSAGE_TOWER_KILL':
                time = int(row['time'])
                i,j = divmod(time, bin)
                team = int(row['team'])-2
                ind = (i*6)+(team*3)
                if ind < len(features):
                    features[ind] += 1
                else:
                    pass
            if row['subtype'] == 'CHAT_MESSAGE_BARRACKS_KILL':
                time = int(row['time'])
                i, j = divmod(time, bin)
                if row['key'] <= 32:  # Powers of Two, Dire->Radiant, Bottom -> Top, Melee-> Ranged.
                    team = 1
                else:
                    team = 0
                ind = (i*6) + (team * 3) + 1
                if ind < len(features):
                    features[ind] += 1
                else:
                    pass
            if row['subtype'] == 'CHAT_MESSAGE_ROSHAN_KILL':
                time = int(row['time'])
                i,j = divmod(time, bin)
                team = int(row['team'])-2
                ind = (i*6)+(team*3) + 2
                if ind < len(features):
                    features[ind] += 1
                else:
                    pass
            if row['subtype'] == 'CHAT_MESSAGE_FIRSTBLOOD':
                if row['player1'] < 5:
                    fb = 0
                else:
                    fb = 1

        self._obj_feat = features
        self._firstblood = fb
        return


    def get_label(self):
        return int(self._label)

    def bit2arr(self, bit_str):  # converts a bit string to an array.
        arr = (",").join(bit_str).split(",")
        for i in range(len(arr)):
            arr[i] = int(arr[i])

        return arr
