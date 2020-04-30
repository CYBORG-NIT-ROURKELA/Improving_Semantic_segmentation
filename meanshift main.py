# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 02:43:55 2020

@author: RajArPatra
"""


from skimage import io
import matplotlib.pyplot as plt
import cv2
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from skimage import io
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import cv2
import numpy as np
from sklearn.cluster import MeanShift
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2lab,rgb2gray,gray2rgb



band=estimate_bandwidth(y) # should be run after running the kde file where it has the variable y.
pix_size=100
ms = MeanShift(band)
rett1=ms.fit_predict(y)
rett=rett1.astype('int8').tolist()
cluster_centers = ms.cluster_centers_
zero=np.zeros(3,)
#for masking the final image
for i in range(len(rett)):
    if rett[i]==0:
      arr=img_arr[i]  
      rett[i]=img_arr[i]  
    else:
        
        rett[i]=zero

rett=np.asanyarray(rett,'uint8')
        
final=rett.reshape(pix_size,pix_size,3).astype('uint8')
combined=np.hstack((final,img))
combined=combined.astype('uint8')

plt.imshow(combined)
    
rett1_re=rett1.reshape(pix_size,pix_size)       
out1 = color.label2rgb(rett1_re, img,kind='avg')
plt.imshow(out1 )