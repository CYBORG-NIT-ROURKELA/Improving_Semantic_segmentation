# -*- coding: utf-8 -*-
"""
Created on Sat May  2 01:14:32 2020

@author: Santanu
"""

import numpy as np

class Improve:
    def __init__(self, deeplab, segments, cluster_type):
        self.deeplab = deeplab
        self.segments = segments
        
        #storing the no. of cluster centers 
        if cluster_type == 'watershed':
            self.a = len(np.unique(segments)) + 1
        else:
            self.a = len(np.unique(segments))
        
        #no. of classes
        self.b = 21
    
    def dispart(self):
        clust_stat = np.zeros((self.a, self.b))

        for i in range(self.deeplab.shape[0]):
            for j in range(self.deeplab.shape[1]):
                clust_stat[self.segments[i, j], self.deeplab[i, j]] += 1
           
        clust_select = np.argmax(clust_stat, axis=1)
        final_seg = np.zeros((self.deeplab.shape[0], self.deeplab.shape[1]))
        
        for i in range(self.deeplab.shape[0]):
            for j in range(self.deeplab.shape[1]):
                final_seg[i, j] = clust_select[self.segments[i, j]]
        
        return final_seg