# 模拟人员类
# author : 陈麒先
# date : 2018-12-13

from functions import ReLU, distanceP2W
import numpy as np
np.seterr(divide='ignore', invalid='ignore')


class Agent(object):
    def __init__(self, position):
        # random initialize a agent
        self.mass = 60.0            # 人的质量
        self.radius = 0.3           # 人的半径
        self.desiredV = 0.8         # 人的期望速度
        self.direction = np.array([0.0, 0.0])  # 人的期望速度方向 （应修改为用A*算法更新）
        self.actualV = np.array([0.0, 0.0])  # 人的实际速度（向量：大小+方向）用 v = v0 + at更新
        self.tau = 0.5   # 即dt仿真时间间隔
        self.pos = position         # 人的位置
        self.dest = np.array([100.0, 10.0])   # 目的地 （出口位置）
        self.bodyFactor = 120000       # 公式中第一项的 K
        self.slideFricFactor = 240000  # 公式中第二项的 k
        self.A = 2000                  # Ai
        self.B = 0.08                  # Bi

    # 以下计算受力带入公式计算

    # 自驱动力
    def adaptVel(self):
        deltaV = self.desiredV * self.direction - self.actualV
        if np.allclose(deltaV, np.zeros(2)):   # 若deltaV 接近0则置0
            deltaV = np.zeros(2)
        return deltaV * self.mass / self.tau

    # 人与人之间的力
    def peopleInteraction(self, other):
        rij = self.radius + other.radius
        dij = np.linalg.norm(self.pos - other.pos)
        nij = (self.pos - other.pos) / dij
        first = (self.A * np.exp((rij - dij) / self.B) +
                 self.bodyFactor * ReLU(rij - dij)) * nij
        tij = np.array([-nij[1], nij[0]])
        deltaVij = (self.actualV - other.actualV) * tij
        second = self.slideFricFactor * ReLU(rij-dij) * deltaVij * tij
        # print("##p2pForce")
        # print(first + second)
        # print("##rij , dij")
        # print(rij , dij)
        # print("@ReLU")
        # print(ReLU(rij - dij))
        return first + second

    # 人与墙之间的力
    def wallInteraction(self, wall):
        ri = self.radius
        diw, niw = distanceP2W(self.pos, wall)  # d 为距离 ，n为方向向量
        first = (self.A * np.exp((ri - diw) / self.B) +
                 self.bodyFactor * ReLU(ri-diw)) * niw
        tiw = np.array([-niw[1], niw[0]])
        second = self.slideFricFactor * \
            ReLU(ri - diw) * (self.actualV * tiw) * tiw
        return first - second
