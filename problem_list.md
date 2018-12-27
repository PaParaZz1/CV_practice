深度学习基础：

1. maxpool的梯度怎么反传
2. 为什么maxpool会work，他和avgpool有什么不同
3. 为什么神经网络要加深层数而不是拓展某一层的隐藏节点数
4. 为什么要使用mini-batch梯度下降，batch大或小各有什么优点
5. 为什么神经网络如果没有激活函数能力不行
6. 设计激活函数时要考虑哪些因素
7. 以全连接层为例，权重应该如何初始化
8. 什么是过拟合和欠拟合，其对应的测试数据的偏差和方差是什么样的
9. 正则化的各种方式（网络角度，loss角度，数据角度）
10. 写出BatchNorm的完整形式
11. （optional）了解LayerNorm, Sync BN, InstanceNorm, GroupNorm, SwitchNorm, SpectrualNorm
12. （optional）了解Dropout的各种变形
13. （optional）了解Data augment的各种类型，以及如何选择合适的数据增强
14. 梯度消失或梯度爆炸的理解
15. L1正则化和L2正则化的效果的不同，原因，什么场景适用L1什么适用L2（几何角度，梯度角度，概率分布角度）
16. L0范数和L无穷范数的形式，为什么我们不用其他Lp范数
17. 理解带动量的SGD
18. （optional）理解RMSprop和Adam
19. 如何调节学习率
20. 简单的超参数搜索策略
21. BN在Inference（test）时如何使用
22. Softmax，sigmoid，tanh各自的形式，推导二分类和多分类的CrossEntropy形式
23. 机器学习中常用的评判标准，accuracy，recall，F1-score，IOU，PR曲线，ROC曲线

卷积神经网络系列

1. Conv1d,conv2d,conv3d手写（基本模式）
2. 卷积相比全连接层的好处
3. 会计算一层卷积之后的feature map大小，给定卷积核大小，能计算需要padding多少才能保持feature map不变
4. 会计算感受野（receptive field）
5. 卷积核大小带来的好处和坏处
6. 非正方形（长方形）卷积核的用处
7. 卷积核为什么常常是奇数正方形
8. 1 * 1卷积的用处
9. 如何用卷积代替池化层
10. Deconv与Upsample+Conv（上采样）
11. 分组卷积
12. 空洞卷积
13. （optional）Deformable Conv
14. VGG
15. Inception系列
16. 残差网络（ResNet），skip connect

pytorch和numpy使用系列

...to be continued

