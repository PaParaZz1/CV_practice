# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 22:41:32 2018

@author: nyz
"""

import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

#t=Image.open('./test/32.jpg').convert('RGB')
'''
t=cv2.imread('./test/65.jpg')
print(t.shape)
gray = cv2.cvtColor(t,cv2.COLOR_BGR2GRAY)
th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

cv2.imwrite('th2.jpg',th2)
w=[x for x in range(32)]
y=th2.sum(axis=0)
print(y.shape)
print(y.mean())
for i in range(1,len(y)-1,1):
    if y[i] < y[i-1] and y[i] < y[i+1]:
        print('min:',i)
t1=t[:,12:19,:]
t1=cv2.resize(t1,(32,32))
cv2.imwrite('t1.jpg',t1)
plt.plot(w,y)
'''
def find_peak(th_sum, min_limit, max_limit):
    threshold = 1.2
    th_slice = th_sum[min_limit:max_limit+1]
    th_mean = th_slice.mean()
    th_max = th_slice.max()
    th_min = th_slice.min()
    if th_max > th_mean * threshold:
        return th_slice.tolist().index(th_max) + min_limit
    if th_min < th_mean / threshold:
        return th_slice.tolist().index(th_min) + min_limit
    return -1


def segment(filename, root='./test/', norm=True, store=False, store_root='./target_fine/'):
    if store and not os.path.exists(store_root):
        os.mkdir(store_root)
    img = Image.open(os.path.join(root + filename)).convert('RGB')
    img_np = np.array(img)
    gray = cv2.cvtColor(img_np,cv2.COLOR_RGB2GRAY)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    th_sum = th.sum(axis=0)
    min_left = 8
    max_left = 12
    min_right = 20
    max_right = 24
    
    left = find_peak(th_sum, min_left, max_left)
    right = find_peak(th_sum, min_right, max_right)
    if left == -1:
        left = min_left
    if right == -1:
        right = max_right
    img_np[:,:left-1] = 0
    img_np[:,right+1:] = 0
    #img_save_np = cv2.resize(img_np[:,left-1:right+1], (32,32))
    img_save_np = img_np
    img_float_np = np.float32(img_np)
    if norm:
        img_slice = img_float_np[:,left-1:right+1]
        print(img_slice.shape)
        img_mean = img_slice.mean(axis=1)
        img_mean = img_mean.mean(axis=0)
        img_std = img_slice.std(axis=1)
        img_std = img_std.std(axis=0)
        img_float_np[:,left-1:right+1] = (img_slice-img_mean) / img_std
    if store:
        img = Image.fromarray(np.uint8(img_save_np))
        img.save(os.path.join(store_root + filename.split('.')[0] + '_seg.jpg'))
    print('end')
    return img_float_np


if __name__ == '__main__':
    out=segment('32.jpg',store=True)
    print(out.shape)
    print(out)
    print(out.mean())
    print(out.max())
    print(out.min())
    #reverse()