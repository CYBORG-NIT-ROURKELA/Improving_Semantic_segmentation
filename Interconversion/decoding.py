# -*- coding: utf-8 -*-
"""
Created on Sat May  2 00:46:33 2020

@author: Santanu
"""

import numpy as np
import torch

class ImageVisualise:
    def __init__(self, pascal_matrix):
        #pascal_matrix is the matrix where element(pixel) stores the class to which it belong
        self.pascal_matrix = pascal_matrix
        
    def RGBImage(self):
        #making 21 colors for 21 different classes
        colors = (torch.as_tensor([i for i in range(21)])[:, None] * torch.tensor([2**25-1, 2**15-1, 2**21-1])%255)
        colors = colors.numpy().astype('uint8')
        
        #creating three channels 
        r = np.zeros_like(self.pascal_matrix).astype(np.uint8)
        g = np.zeros_like(self.pascal_matrix).astype(np.uint8)
        b = np.zeros_like(self.pascal_matrix).astype(np.uint8)
        
        for i in range(21):
            idx = self.pascal_matrix == i
            
            #assigning the color to the channels
            r[idx] = colors[i, 0]
            g[idx] = colors[i, 1]
            b[idx] = colors[i, 2]
        
        #stacking the 3 channels to get rgb image
        rgb_image = np.stack([r, g, b], axis=2)
        return rgb_image
