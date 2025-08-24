# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 15:55:05 2019

@author: techj
"""

class PriorityQueue:
    def __init__(self, size):
        self.size = size
        self.queue = []
    def insert(self, word, spam):
        interesting = abs(0.5 - spam)
        index = 0
        while index < len(self.queue) and interesting < self.queue[index][0]:
            index += 1
        if index < self.size and not (index < len(self.queue) and word == self.queue[index][1]):
            self.queue.insert(index, (interesting, word, spam))
        if len(self.queue) > self.size:
            self.queue = self.queue[:-1]
    def getSpamicityList(self):
        spamicitylist = []
        for double in self.queue:
            spamicitylist.append(double[2])
        return spamicitylist