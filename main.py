# -*- coding: utf-8 -*-
"""
Created on Sat May  2 13:20:54 2020

@author: Santanu
"""
import time
start_time = time.time()

import os 

images = []
images_dir = "C:/Desktop/pascal_mini_2/JPEGImages/"
for _, _, files in os.walk(images_dir):
    for x in files:
        if x.endswith(".jpg") == True:
            images.append(x)

masks = []
mask_dir = "C:/Desktop/pascal_mini_2/SegmentationClass/"
for _, _, files in os.walk(mask_dir):
    for y in files:
        if y.endswith(".png") == True:
            masks.append(y)

from torchvision import models
deeplab_model = models.segmentation.deeplabv3_resnet101(pretrained = 1)
deeplab_model = deeplab_model.cuda()
deeplab_model.eval()

from PIL import Image
import torch
import torchvision.transforms as T

t = T.Compose([T.Resize(256),
               T.CenterCrop(224),
               T.ToTensor(),
               T.Normalize(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])])
    
deeplab = []    
for i in range(len(images)):
    x = Image.open(images_dir + images[i])
    y = t(x).unsqueeze(0)
    y = y.cuda()
    with torch.no_grad():
        out = deeplab_model(y)['out']
    seg_img = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
    deeplab.append(seg_img)

import numpy as np

def decode(pmatrix):
    colors = (torch.as_tensor([i for i in range(21)])[:, None] * torch.tensor([2**25-1, 2**15-1, 2**21-1])%255)
    colors = colors.numpy().astype('uint8')
    
    r = np.zeros_like(pmatrix).astype(np.uint8)
    g = np.zeros_like(pmatrix).astype(np.uint8)
    b = np.zeros_like(pmatrix).astype(np.uint8)
    
    for i in range(21):
        idx = pmatrix == i
        r[idx] = colors[i, 0]
        g[idx] = colors[i, 1]
        b[idx] = colors[i, 2]
    
    image = np.stack([r, g, b], axis=2)    
    return image

def encode(mask_loc):
    mask_image = Image.open(mask_loc)
    
    t = T.Compose([T.Resize(256),
                T.CenterCrop(224)])
    mask_image_resized = t(mask_image)
    
    mask = np.asarray(mask_image_resized).astype('int16')
    
    label = np.where(mask == 255, 0, mask)
    return label

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed

def cluster(image_loc):
    image = Image.open(image_loc)
    
    t = T.Compose([T.Resize(256),
                   T.CenterCrop(224)])
    image_resized = t(image)
    
    img = np.array(image_resized)
    
    img_fz = felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    img_slic = slic(img, n_segments=250, compactness=10, sigma=1)
    img_quick = quickshift(img, kernel_size=1, max_dist=6, ratio=0.5,sigma=0)
    gradient = sobel(rgb2gray(img))
    img_watershed = watershed(gradient, markers=250, compactness=0.001)
    
    return img_fz, img_slic, img_quick, img_watershed

def dispart(deeplab, segments, cluster_type):
    if cluster_type == 'watershed':
        a = len(np.unique(segments)) + 1
    else:
        a = len(np.unique(segments))
    b = 21
    
    clust_stat = np.zeros((a, b))
    
    for i in range(deeplab.shape[0]):
        for j in range(deeplab.shape[1]):
            clust_stat[segments[i,j],deeplab[i,j]]+=1
    
    clust_select = np.argmax(clust_stat, axis=1)
    final_seg = np.zeros((deeplab.shape[0], deeplab.shape[1]))
    
    for i in range(deeplab.shape[0]):
        for j in range(deeplab.shape[1]):
            final_seg[i, j] = clust_select[segments[i, j]]
    
    return final_seg.astype('int16')

def compute_miou(actual, pred):
    a = actual
    a = a.reshape((50176,))
    a_count = np.bincount(a, weights = None, minlength = 21) # A
    
    b = pred
    b = b.reshape((50176,))
    b_count = np.bincount(b, weights = None, minlength = 21) # B
    
    c = a * 21 + b
    cm = np.bincount(c, weights = None, minlength = 441)
    cm = cm.reshape((21, 21))
    
    Nr = np.diag(cm) # A ⋂ B
    Dr = a_count + b_count - Nr # A ⋃ B
    individual_iou = Nr/Dr
    miou = np.nanmean(individual_iou)
    
    return miou

label =[]
segments_fz = []
segments_slic = []
segments_quick = []
segments_watershed = []
final_seg_fz = []
final_seg_slic = []
final_seg_quick = []
final_seg_watershed = []
miou_deeplab = []
miou_fz = []
miou_slic = []
miou_quick = []
miou_watershed = []

for i in range(len(images)):
    #encoding the ground truth mask
    temp_label = encode(mask_dir + masks[i])
    label.append(temp_label)
    
    #applying 4 different types of clustering on each image
    a, b, c, d = cluster(images_dir + images[i])
    segments_fz.append(a)
    segments_slic.append(b)
    segments_quick.append(c)
    segments_watershed.append(d)
    
    #improving the results of deeplab by disparting
    seg_fz = dispart(deeplab[i], a, 'fz')
    seg_slic = dispart(deeplab[i], b, 'slic')
    seg_quick = dispart(deeplab[i], c, 'quick')
    seg_watershed = dispart(deeplab[i], d, 'watershed')
    final_seg_fz.append(seg_fz)
    final_seg_slic.append(seg_slic)
    final_seg_quick.append(seg_quick)
    final_seg_watershed.append(seg_watershed)
    
    #mean IoU calculation
    deeplab_miou = compute_miou(temp_label, deeplab[i])
    fz_miou = compute_miou(temp_label, seg_fz)
    slic_miou = compute_miou(temp_label, seg_slic)
    quick_miou = compute_miou(temp_label, seg_quick)
    watershed_miou = compute_miou(temp_label, seg_watershed)
    miou_deeplab.append(deeplab_miou)
    miou_fz.append(fz_miou)
    miou_slic.append(slic_miou)
    miou_quick.append(quick_miou)
    miou_watershed.append(watershed_miou)
 
d = 0.0
f = 0.0
s = 0.0
q = 0.0
w = 0.0
n = len(images)

for i in range(n):
    d += miou_deeplab[i]
    f += miou_fz[i]
    s += miou_slic[i]
    q += miou_quick[i]
    w += miou_watershed[i]
    
d = d/n
f = f/n
s = s/n
q = q/n
w = w/n

print('Average result of deeplab :', d)
print('Average result of disparting with fz clustering :', f)
print('Average result of disparting with slic clustering :', s)
print('Average result of disparting with quick clustering :', q)
print('Average result of disparting with watershed clustering :', w)

print('Time taken :', time.time()-start_time, 'second')

print('----------------------------------Mean IoU table---------------------------------------')
print('Sl No. \tdeeplab \tusing fz \tusing slic \tusing quick \tusing watershed')
print('---------------------------------------------------------------------------------------')

for i in range(n):
    a = miou_deeplab[i]
    b = miou_fz[i]
    c = miou_slic[i]
    d = miou_quick[i]
    e = miou_watershed[i]
    print("%d\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f"%(i+1, a, b, c, d, e))



    
    
    
    
    
    