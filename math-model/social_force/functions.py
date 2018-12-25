# coding=utf-8
# 功能函数
# author : 陈麒先
# date : 2018-12-14

import numpy as np


# ReLU 即函数g
def ReLU(x):
    if x > 0:
        return x
    return 0


# 向量v的标准化，化成单位向量
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v/norm


# person to wall距离（点到线段距离）
def distanceP2W(point, wall):
    p0 = np.array([wall[0], wall[1]])
    p1 = np.array([wall[2], wall[3]])
    d = p1-p0
    ymp0 = point-p0
    t = np.dot(d, ymp0) / np.dot(d, d)
    if t <= 0.0:
        dist = np.sqrt(np.dot(ymp0, ymp0))
        cross = p0 + t * d
    elif t >= 1.0:
        ymp1 = point - p1
        dist = np.sqrt(np.dot(ymp1, ymp1))
        cross = p0 + t * d
    else:
        cross = p0 + t * d
        dist = np.linalg.norm(cross - point)
    npw = normalize(cross - point)
    return dist, npw
