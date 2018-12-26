# coding=utf-8
import numpy as np
from heapq import heappush, heappop
from functions import normalize

# H函数，计算点neighbor 到 goal的manhattan距离


def heuristic_cost_estimate(neighbor, goal):
    x = neighbor[0] - goal[0]
    y = neighbor[1] - goal[1]
    return abs(x) + abs(y)

# 计算a , b 两点的欧氏距离


def dist_between(a, b):
    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

# 更新路径


def reconstruct_path(came_from, current):
    path = [current]           # current即为goal
    while current in came_from:
        current = came_from[current]   # 按照came_from中的节点信息复原出路径
        path.append(current)
    return path


# astar function returns a list of points (shortest path)
def astar(array, start, goal):
    # print(array)
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1),
                  (1, -1), (-1, 1), (-1, -1)]  # 8个方向

    close_set = set()  # close list
    came_from = {}     # 路径集（记录最优路径节点）
    gscore = {start: 0}  # g函数字典，值为到起点的距离 ，key为一个点坐标
    # f函数字典，f = g + h,初始化只有起点的h值
    fscore = {start: heuristic_cost_estimate(start, goal)}

    openSet = []     # open list 建立一个常见的堆结构
    heappush(openSet, (fscore[start], start))   # 往堆中插入一条新的值 ，内部存按堆排序的f升序堆

    # while openSet 非空
    while openSet:
        # current := the node in openSet having the lowest fScore value
        current = heappop(openSet)[1]   # 从堆中弹出fscore最小的节点 （heappop函数实现）

        # 循环终止条件
        if current == goal:             # 当openlist包含目的地节点时，返回path
            # 根据came_from字典（k-v对的v指向父节点）生成path并返回
            path = reconstruct_path(came_from, current)
            length = len(path)
            direct = np.array(
                [path[length-2][0] - path[length-1][0], path[length-2][1] - path[length-1][1]])
            return normalize(direct)  # 返回的是当前的从当前位置到目的地的速度方向向量
        close_set.add(current)  # 把当前节点移入close list中

        for i, j in directions:      # 对当前节点的 8 个相邻节点一一进行检查
            neighbor = current[0] + i, current[1] + j   # 相邻节点的计算
            # print("@current")
            # print(current)
            # print("@neighbor")
            # print(neighbor)
            # 判断节点是否在地图范围内，并判断是否为障碍物
            if 0 <= neighbor[0] < array.shape[0]:     # 地图范围内判断
                if 0 <= neighbor[1] < array.shape[1]:  # 地图范围内判断
                    if array[neighbor[0]][neighbor[1]] == 1:   # 1为障碍物，有障碍物判断
                        continue  # 跳过障碍物
                else:
                    # array bound y walls
                    continue  # 跳过超出地图范围
            else:
                # array bound x walls
                continue  # 跳过超出地图范围

            # Ignore the neighbor which is already evaluated.
            if neighbor in close_set:
                continue  # 跳过已进入 Closelist 的节点

            #  计算经过当前节点到达相邻节点的g值，用于比较是否更新
            tentative_gScore = gscore[current] + dist_between(current, neighbor)

            # 如果当前节点的相邻节点不在open list中，将其加入到open list当中
            # Discover a new node
            if neighbor not in [i[1] for i in openSet]:
                heappush(openSet, (fscore.get(neighbor, np.inf), neighbor))
            # 若不是更优的解（g不具有更小值）则跳过该节点
            # This is not a better path.
            elif tentative_gScore >= gscore.get(neighbor, np.inf):
                continue
            # 若未跳过，则该节点为经过的路径，修改came_from列表
            # This path is the best until now. Record it!
            came_from[neighbor] = current  # 相邻节点父节点指向当前节点
            gscore[neighbor] = tentative_gScore
            fscore[neighbor] = tentative_gScore + \
                heuristic_cost_estimate(neighbor, goal)

    return False


if __name__ == "__main__":
    nmap = np.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    path = astar(nmap, (0, 1), (2, 3))

    print(path)
