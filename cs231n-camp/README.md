# cs231n training camp 

## 课程资料
1. [课程主页](http://cs231n.stanford.edu/)  
2. [英文笔记](http://cs231n.github.io/)  
3. [中文笔记](https://zhuanlan.zhihu.com/p/21930884)  
4. [课程视频](https://www.bilibili.com/video/av17204303/)  
5. [环境配置](https://github.com/sharedeeply/DeepLearning-StartKit)  
6. [作业链接](https://github.com/sharedeeply/cs231n-camp/tree/master/assignment/assignment1)  
7. [作业参考](https://github.com/sharedeeply/cs231n-assignment-solution)  
8. [AWS 云服务器配置](https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch/blob/master/aws.md)   
**注: 云服务器并不是强制要求的，而且国外的服务器会比较卡，考虑到阿里云等国内的服务器比较贵，所以推荐大家使用本地的电脑**


#### 🔥🔥一些重要的资源：

1. [廖雪峰python3教程](https://www.liaoxuefeng.com/article/001432619295115c918a094d8954bd493037b03d27bf9a9000)
2. [github教程](https://www.liaoxuefeng.com/wiki/0013739516305929606dd18361248578c67b8067c8c017b000)
3. [深度学习的学习路线](https://github.com/L1aoXingyu/Roadmap-of-DL-and-ML/blob/master/README_cn.md)和[开源深度学习课程](http://www.deeplearningweekly.com/blog/open-source-deep-learning-curriculum/)
4. [mxnet/gluon 教程](https://zh.gluon.ai/)
5. [我的知乎专栏](https://zhuanlan.zhihu.com/c_94953554)和[pytorch教程](https://github.com/L1aoXingyu/code-of-learn-deep-learning-with-pytorch)
6. [官方pytorch教程](https://pytorch.org/tutorials/)和一个比较好的[教程](https://github.com/yunjey/pytorch-tutorial)
7. [tensorflow教程](https://github.com/aymericdamien/TensorFlow-Examples)

上面是本次训练营经常需要用到的网页，所以顶置便于大家查询

## 前言
对于算法工程师，不同的人的认知角度都是不同的，我们通过下面三个知乎的高票回答帮助大家了解算法工程师到底需要做什么样的事，工业界需要什么样的能力

[从今年校招来看，机器学习等算法岗位应届生超多，竞争激烈，未来 3-5 年机器学习相关就业会达到饱和吗?](https://www.zhihu.com/question/66406672/answer/317489657)

[秋招的 AI 岗位竞争激烈吗?](https://www.zhihu.com/question/286925266/answer/491117602)

[论算法工程师首先是个工程师之深度学习在排序应用踩坑总结](https://zhuanlan.zhihu.com/p/44315278)

## 学习安排
### Week 1
1. 计算机视觉综述  
**slides:** [lecture01](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture01.pdf)
- 观看视频 p1 和 p2 热身，了解计算机视觉概述以及历史背景
- 观看 p3 了解整门课程的大纲

2. 学习数据驱动的方法和 KNN 算法和线性分类器[上]  
**slides:** [lecture02](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture02.pdf) 
- 观看视频 p4 p5 和 p6
- 学习 [图像分类笔记上下](https://zhuanlan.zhihu.com/p/20894041?refer=intelligentunit) 和 [线性分类笔记上](https://zhuanlan.zhihu.com/p/20918580?refer=intelligentunit)

**作业:**   
1. [阅读 python 和 numpy 教程](https://zhuanlan.zhihu.com/p/20878530?refer=intelligentunit)和[代码](https://github.com/sharedeeply/cs231n-camp/blob/master/tutorial/python_numpy_tutorial.ipynb)写一个矩阵的类，实现矩阵乘法，只能使用 python 的类(class)和列表(list)
2. 完成assignment1 中的 `knn.ipynb`


### Week2
1. 学习线性分类器[中 下], 损失函数和优化器  
   **slides:** [lecture03](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture03.pdf)
- 观看视频 p7 和 p8，了解更多关于线性分类器，损失函数以及优化器的相关知识
- 学习[线性分类笔记中下](https://zhuanlan.zhihu.com/p/20945670?refer=intelligentunit)和[最优化笔记](https://zhuanlan.zhihu.com/p/21360434?refer=intelligentunit)，了解 SVM 和梯度下降法

**作业:**
1. 简述 KNN 和线性分类器的优劣
2. (可选)学习[矩阵求导](https://zhuanlan.zhihu.com/p/25063314)的方法
2. 完成assignment1 中 `svm.ipynb`


### Week3
1. 神经网络初步  
**slides:** [lecture04](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture04.pdf)

- 观看视频 p9 和 p10
- 学习[反向传播算法的笔记](https://zhuanlan.zhihu.com/p/21407711?refer=intelligentunit)和反向传播算法的[数学补充](http://cs231n.stanford.edu/handouts/derivatives.pdf)和[例子](http://cs231n.stanford.edu/handouts/linear-backprop.pdf) 
可选项：[反向传播算法的博客](http://colah.github.io/posts/2015-08-Backprop/)

**作业:**  
1. 理解并推导反向传播算法
2. 完成 assignment1 中的 `softmax.ipynb` 和 `two_layer_net.ipynb`

### Week4
1. 学习 [pytorch基础](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) 
2. 了解 kaggle 比赛[房价预测](https://www.kaggle.com/c/house-prices-advanced-regression-techniques), 并学习[模板代码](https://github.com/L1aoXingyu/kaggle-house-price)

**作业:**  
1. 完成 assignment1 中  和 `features.ipynb`
2. 修改房价预测的代码，并提交kaggle查看得分


### Week5
1. 卷积神经网络初步
**slides:** [lecture05](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture05.pdf)

- 观看视频 p11, p12 和 p13，学习 CNN 中的卷积层和池化层
- 学习[卷积神经网络笔记](https://zhuanlan.zhihu.com/p/22038289?refer=intelligentunit)

**作业:**  
1. 完成 assignment2 中 `FullyConnectedNets.ipynb` 和 `BatchNormalization.ipynb`
2. 思考一下卷积神经网络对比传统神经网络的优势在哪里？为什么更适合处理图像问题
3. 了解和学习深度学习中的[normalization方法](https://zhuanlan.zhihu.com/p/33173246)


### Week6
1. 如何更好的训练网络(上)  
**slides:** [lecture06](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture06.pdf)

- 观看视频 p14, p15，学习训练神经网络中的激活函数，初始化和正则化方法
- 学习[神经网络笔记1](https://zhuanlan.zhihu.com/p/21462488?refer=intelligentunit)和[神经网络笔记2](https://zhuanlan.zhihu.com/p/21560667?refer=intelligentunit)

**作业:**  
1. 完成 assignment2 中 `Dropout.ipynb` 和 `ConvolutionNetworks.ipynb`
2. 了解和学习 `fast conv` 和 `fast pooling` 是如何实现的


### Week7
1. 如何更好的训练网络(下)  
**slides:** [lecture07](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture07.pdf)

- 观看视频 p16, p17 和 p18, 了解训练神经网络中更多的标准化方法以及更多的学习率更新策略
- 学习[神经网络笔记3](https://zhuanlan.zhihu.com/p/21741716?refer=intelligentunit)

**作业:**  
1. 完成 assignment2 中 PyTorch CIFAR10 `PyTorch.ipynb`
2. 学习深度学习中各种优化算法的[总结](https://zhuanlan.zhihu.com/p/22252270)
3. 打kaggle比赛 [cifar10](https://www.kaggle.com/c/cifar-10), [模板代码]()

### Week8
1. 深度学习框架介绍   
**slides:** [lecture08](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture08.pdf)

- 观看视频 p19，了解深度学习的主流框架

2. 经典的网络结构结构  
**slides:** [lecture09](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture09.pdf)

- 观看视频 p20，了解目前计算机视觉 state of the art 的网络结构

**作业:**  
1. 根据前面学的知识，完成 ai challenger[农作物灾害监测](https://challenger.ai/competition/pdr2018?type=myteam)的比赛，并提交成绩，[模板代码]()


### Week9
1. 循环神经网络与语言模型  
**slides:** [lecture10](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture10.pdf)

- 观看视频 p21, p22 和 p23，了解循环神经网络，LSTM以及图片文字生成的方法

**作业:**  
1. 完成 assignment3 中的 `RNN_Captioning.ipynb` 和 `LSTM_Captioning.ipynb` 
2. 根据[blog](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)更进一步了解RNN，写出 LSTM 和 GRU 的公式

### Week10
1. 检测与分割  
**slides:** [lecture11](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture11.pdf)

- 观看视频 p24，p25 和 p26，了解检测和分割的任务介绍  
- 阅读 [SSD](https://arxiv.org/abs/1512.02325) 和 [Faster RCNN](https://arxiv.org/abs/1506.01497) 的论文

**作业:**  
1. 学习 SSD 的模板代码，跑 voc 数据集
2. 学习 FCN 的模板代码，跑 voc 数据集

### Week11
1. 生成对抗网络  
**slides:**  [lecture12](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture12.pdf)

- 观看视频 p29，p30 和 p31，了解变分自动编码器和生成对抗网络

2. 卷积的可视化理解  
**slides:** [lecture13](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture13.pdf)

- 观看视频 p27 和 p28，探索卷积网络背后的原理，学习 deep dream 和 style transfer 等有趣的应用

**作业:**  
1. 完成 assignment3 中的 `GANs-PyTorch.ipynb`
2. 完成 assignment3 中的 `NetworkVisualization-PyTorch.ipynb` 和 `StyleTransfer-PyTorch.ipynb`


### Week12
1. 深度强化学习
**slides:** [lecture14](http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture14.pdf)

- 观看视频 p32 和 p33，了解深度强化学习中的 q-learning 和 actor-critic 算法

**作业:**  
1. 学习使用 Deep Q-Network 玩 cartpole 的游戏