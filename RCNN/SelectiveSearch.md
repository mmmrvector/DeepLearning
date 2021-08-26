## Selective Search ##

### 背景介绍 ####

目标检测问题比图像分类问题复杂，很重要的因素在于图像中可能存在多个物体需要分别定位和分类。

在给对象做Region Proposal的时候，不能通过单一的策略来区分，需要充分考虑图像物体的多样性。

+ Exhaustive Search

一开始是穷举法，遍历每一个像素，导致搜索的范围很大，计算量也很大，一般使用HOG(Histogram of Oriented Gradient) + SVM

+ Segmentation

+ Other sampling Strategies

### Selective Search ###

基于Region Proposal方法的基础上，作者提出了Selective Search。这个方法主要有三个优势：捕捉不同尺度、多样化、快速计算。

Selective Search算法主要包括两个内容：Hierarchical Grouping Algorithm & Diversification Strategies

+ **Hierarchical Grouping Algorithm**

图像中区域特征比像素更具代表性，作者使用**Felzenszwalb and Huttenlocher**的方法产生图像初始区域，使用贪心算法对区域进行迭代分组：

1. 计算所有邻近区域之间的相似性
2. 两个最相似的区域被组合在一起
3. 计算合并区域和相邻区域的相似度
4. 重复2、3过程，知道图像变成一个地区

算法简单描述：

**输入**：图片（三通道）

**输出**：物体位置的可能结果L

1. 使用 Felzenszwalb and Huttenlocher提出的方法得到初始分割区域R={r1,r2,…,rn}；
2. 初始化相似度集合S=∅；
3. 计算两两相邻区域之间的相似度，将其添加到相似度集合S中；
4. 从集合S中找出，相似度最大的两个区域 ri 和rj，将其合并成为一个区域 rt，从集合中删去原先与ri和rj相邻区域之间计算的相似度，计算rt与其相邻区域（与ri或rj相邻的区域）的相似度，将其结果加入到相似度集合S中。同时将新区域 rt 添加到区域集合R中；
5. 获取每个区域的Bounding Boxes L，输出物体位置的可能结果L。

+ **Diversification Strategies**
这个部分讲述作者提到的多样性的一些策略，使得抽样多样化，主要有下面三个不同方面：
  
1. 利用各种不同不变性的色彩空间
2. 采用不同的相似性度量
3. 通过改变起始区域。

#### Similarity Measuers 相似度衡量 ####

+ 颜色相似度衡量

+ 纹理相似度衡量

+ 尺度相似度衡量

+ 形状重合度衡量

+ 最终相似度衡量



### 参考文献 ###
[1]. 《理解Selective Search》https://zhuanlan.zhihu.com/p/39927488