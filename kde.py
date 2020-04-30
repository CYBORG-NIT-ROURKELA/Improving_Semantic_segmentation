# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:23:32 2020

@author: RajArPatra
"""


from skimage import io
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import cv2
import numpy as np
from sklearn.cluster import MeanShift
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.color import rgb2lab,rgb2gray,gray2rgb


img=io.imread("dog.jfif",as_gray=False) #
pix_size=100 #  dimensions of the image
img=cv2.resize(img,(pix_size,pix_size))
img_arr=img.reshape(-1,3)

img_gray=rgb2gray(img)
img_gray_arr=img_gray.reshape(-1)

print("org_img",img)
plt.imshow(img)
#img=rgb2lab(img)
img1=img.reshape(-1,3)
plt.imshow(img)

kde= KernelDensity(kernel='gaussian', bandwidth=0.1).fit(img1)

z=kde.score_samples(img1)
print("kde_img",z.reshape(pix_size,pix_size))
y=z[:,None]
z1=z.reshape(pix_size,pix_size)

#segments_quick = quickshift(z1, kernel_size=3, max_dist=6, ratio=0.5)
#plt.imshow(mark_boundaries(z1, segments_quick))
#plt.imshow(mark_boundaries(img, segments_quick))

#z2=np.hstack((img,z1))
plt.imshow(z1)