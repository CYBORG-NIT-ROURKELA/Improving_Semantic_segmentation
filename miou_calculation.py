# -*- coding: utf-8 -*-
"""
Created on Fri May  1 23:07:35 2020

@author: Santanu
"""

import numpy as np

class ConfusionMatrix:
    def __init__(self, pred, actual):
        self.pred = pred
        self.actual = actual
    
    def construct(self):
        
        assert(self.pred.shape == self.actual.shape)
        assert(self.pred >= 0 and self.actual >= 0)
        assert(self.pred < 21 and self.actual < 21)
        
        #-------------converting into 1d array and then finding the frequency of each class------------- 
        self.pred = self.pred.reshape((50176,))
        #storing the frequency of each class present in the predicted mask
        self.pred_count = np.bincount(self.pred, weights = None, minlength = 21) # A
        
        self.actual = self.actual.reshape((50176,))
        #storing the frequency of each class present in the actual mask
        self.actual_count = np.bincount(self.actual, weights = None, minlength = 21) # B
        #-----------------------------------------------------------------------------------------------
        
        '''there are 21 classes but altogether 21x21=441 possibilities for every pixel
        for example, a pixel may actually belong to class '4' but may be predicted to be in class '3'
        So every pixel will have two features, one of which is actual and the other predicted
        To store both the details, we assign the category to which it belong
        Like in the above mentioned example the pixel belong to category 4-3'''
        
        #store the category of every pixel
        temp = self.actual * 21 + self.pred
        
        #frequency count of temp gives the confusion matrix 'cm' in 1d array format
        self.cm = np.bincount(temp, weights = None, minlength = 441)
        #reshaping the confusion matrix from 1d array to (no.of classes X no. of classes)
        self.cm = self.cm.reshape((21, 21))
        
        #the diagonal values of cm correspond to those pixels which belong to same class in both predicted and actual mask
        self.Nr = np.diag(self.cm) # A ⋂ B
        self.Dr = self.pred_count + self.actual_count - self.Nr # A ⋃ B
        
    def computeMiou(self):
        individual_iou = self.Nr / self.Dr # (A ⋂ B)/(A ⋃ B)
        miou = np.nanmean(individual_iou) # nanmean is used to neglect 0/0 case which arise due to absence of any class
        return miou
