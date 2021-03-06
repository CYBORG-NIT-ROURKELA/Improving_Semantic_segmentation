# -*- coding: utf-8 -*-
"""
Created on Sat May  2 18:07:28 2020

@author: Asus
"""
import matplotlib.pyplot as plt

t = T.Compose([T.Resize(256),
               T.CenterCrop(224)])

rh = np.zeros((224, 10)).astype('int16')
gh = np.zeros((224, 10)).astype('int16')
bh = np.zeros((224, 10)).astype('int16')

rh = np.where(rh == 0, 255, 0)
gh = np.where(gh == 0, 255, 0)
bh = np.where(bh == 0, 255, 0)

hblank = np.stack([rh, gh, bh], axis=2)

rv = np.zeros((10, 926)).astype('int16')
gv = np.zeros((10, 926)).astype('int16')
bv = np.zeros((10, 926)).astype('int16')

rv = np.where(rv == 0, 255, 0)
gv = np.where(gv == 0, 255, 0)
bv = np.where(bv == 0, 255, 0)

vblank = np.stack([rv, gv, bv], axis=2)

rows = []
from skimage.color import label2rgb

for i in range(n):
    img = Image.open(images_dir + images[i])
    img = t(img)
    i#mg1 = np.array(img)
    a = decode(label[i])
    b = decode(deeplab[i])
    c = decode(final_seg_quick[i])
    each_row = np.hstack((img, hblank, a, hblank, b, hblank, c))
    rows.append(each_row)

image = vblank
for i in range(len(rows)):
    if i in number:
        image = np.vstack((image, rows[i], vblank))

image2 = vblank
for i in range(n):
    img = Image.open(images_dir + images[i])
    img = t(img)
    if i in number:
        image2 = np.vstack((image2, img, vblank))
    
overlay_image = np.hstack((image2, hblank, (image * 255).astype(np.uint8)))

cv2.imwrite("overlay_collage_2nd_part.jpg", image)
temp = cv2.imread(image)
cv2.imshow("abc", image)
    


