# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:00:32 2020

@author: Asus
"""
number = []
for i in range(n):
    if max(miou_deeplab[i], miou_fz[i], miou_slic[i], miou_quick[i], miou_watershed[i]) == miou_quick[i]:
        number.append(i)
print("no. of cases where quick-shift clustering improves the result", len(number))