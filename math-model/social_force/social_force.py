# 社会力仿真主函数
# author : Chen Qixian
# Date : 2018-12-13 

import numpy as np
import random
np.seterr(divide='ignore',invalid='ignore')
from person import *
from functions import *
from A_star import *

# 定义一些全局变量
AGENTNUM = 5         # 人数设置
ROOMSIZE = 10		 # 房间大小设置（假设正方形房间）
ITERNUM  = 2		 # 迭代次数
agents = []
agent_pos = []
obstacles = []
exits = []

# 初始化地图
MAP = np.zeros((ROOMSIZE , ROOMSIZE))
for i in range(ROOMSIZE):
	MAP[0][i] = 1
	MAP[i][0] = 1
	MAP[ROOMSIZE - 1][i] = 1
	MAP[i][ROOMSIZE - 1] = 1

# 初始化障碍物
for i in range(ROOMSIZE):
	for j in range(ROOMSIZE):
		if MAP[i][j] == 1:
			obstacles.append([i,j])

# 为地图添加出口
exit = [0,5]
exits.append(exit)


MAP[exit[0]][exit[1]] = 0

wall = [3.33, 3.33, 29.97, 3.33]

# 初始化墙面（构造场景）
walls = []
wall1 = [0,0,50,0]
wall2 = [0,0,0,50]
wall3 = [0,50,50,50]
wall4 = [50,0,50,50]
walls.append(wall)
walls.append(wall1)
walls.append(wall2)
walls.append(wall3)
walls.append(wall4)

# 初始化人群（随机位置）
for i in range(AGENTNUM):
	point = [random.random() * ROOMSIZE , random.random() * ROOMSIZE]
	pp =[int(point[0]) , int(point[1])]
	while (pp in exits) or (pp in obstacles) or (pp in agent_pos):  #防止随机生成的行人位置相互重叠，和与障碍物重叠
		point = [random.random() * ROOMSIZE , random.random() * ROOMSIZE]
		pp = [int(point[0]) , int(point[1])]
	agent_pos.append(point)
	position = np.array(point)
	agent = Agent(position)
	agents.append(agent)
	#obstacles = np.array(obstacles)
	#exits = np.array(exits)

# 循环ITERNUM次更新人的状态
for i in range(ITERNUM):
	# 更新每一个人的状态
	for idx , ai in enumerate(agents):
		# 获得人的初始速度，位置
		v0 = ai.actualV
		p0 = ai.pos
		#ai.direction = normalize(ai.dest - ai.pos)  # 期望方向et，为从当前位置指向目的地的单位向量

		# 调用A*算法获取人在当前位置的预期方向
		start = (int(p0[0]) , int(p0[1]))
		#print(MAP)
		#print(start)
		#print(tuple(exit)) 
		ai.direction = astar(MAP , start , tuple(exit))
		#print(ai.direction)
		adaptForce = ai.adaptVel() # 计算自驱动力
		p2pForce = 0 
		w2pForce = 0

		# 计算人与人之间作用力（sigma(fij))
		for idx_other , a_other in enumerate(agents):
			if idx == idx_other:
				continue
			p2pForce += ai.peopleInteraction(a_other)

		# 计算人与墙之间作用力（sigma(fiw))
		for wall in walls:
			w2pForce += ai.wallInteraction(wall)

		# 计算合力
		sumForce = adaptForce + p2pForce + w2pForce
		# 加速度
		accumu = sumForce / ai.mass
		# 更新速度
		ai.actualV = v0 + accumu * ai.tau
		# 更新位移
		ai.pos = p0 + v0 * ai.tau + 0.5 * accumu * (ai.tau ** 2)
		print("加速度： " ,accumu ,"当前实际速度：", ai.actualV,"位置：" , ai.pos)
	print("====================")

print(MAP)