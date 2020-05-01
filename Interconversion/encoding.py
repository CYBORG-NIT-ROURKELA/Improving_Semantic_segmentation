# -*- coding: utf-8 -*-
"""
Created on Fri May  1 22:13:16 2020

@author: Santanu
"""

import numpy as np

class PascalMatrix:
    def __init__(self, img):
        # img(224, 224) is in H x W x 3 format
        self.img = img
        
    def classify(self):
        #defining colors of individual classes
        colors = np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                             [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                             [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                             [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                             [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                             [0, 64, 128]])
    
        colors = colors.astype('int16')        
        self.img = self.img.astype('int16')
        
        #creating a label matrix to store the labels of every pixels
        label = np.zeros((224, 224), dtype = int)
        
        #based on the color of pixels they are classified into different classes
        for a,b in enumerate(colors):
            label[np.where(np.all(self.img == b, axis = -1))[:2]] = a
            
        label = label.astype('int16')
        return label
