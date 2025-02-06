## 模式识别与机器学习实验一：logisitic regression

### 实验任务：

1. 对提供的两组数据 *logistic_data1.csv* , *logisitic_data2.csv*通过逻辑回归进行分类，找出决策面。

   其中，数据1只需找出线性决策面即可。数据2需使用$x_0$,$x_1$两个特征构造高维特征，并分别尝试无正则化和加入L2正则项来查找决策面，对比有无正则项时的分类难度和结果。绘图时可根据参考代码中的画图代码进行绘制。

2. 对提供的鸢尾花数据集通过逻辑回归进行分类。在训练集*iris_train.csv*上训练决策面，并在测试集*iris_test.csv*上执行分类任务，得到准确率。

### 实验要求：

1. 请基于python，手动实现logistic regression过程。切莫使用如`sklearn.LogisiticRegression()`之类的第三方实现，否则**此实验成绩无效**。

1. 请对 *logistic_data1.csv*  , *logisitic_data2.csv*通过给出的画图代码示例绘出包含数据分类结果和决策面的图像。

   如果*data2*难以绘制决策面，只给出分类结果和真实类别的对比图也可以。同时请对*data2*有无正则项的分类结果进行对比，并对结果给出你的解释。

1. 请于9月25日前（两周时间）以邮件附件的形式向邮箱prmlhit2024@163.com提交本实验的实验报告，其命名格式为**lab1-学号-姓名**。报告没有模板，对格式也无任何要求，清楚明白即可。请重点介绍实验任务过程和实验结果，避免粘贴大段代码。注意，请不要剽窃他人的报告或代码。

1. 关于代码验收的时间与方式，请以助教的通知为准。

### 实验数据：

所需的实验数据位于群文件的 *lab1.zip* 中的*/code*路径下，请自行下载。

### 需要的准备：

1. 需要一台能稳定运行的电脑。请不要询问如“我在安装python后电脑怎么卡死了”、“QQ群中下载的文件存储在哪里”或是“为什么我打不开实验文档”之类的问题。
2. 请自行下载python，并配置conda环境。conda环境的配置详见https://www.bilibili.com/read/cv20976411/
2. 本次实验不需要torch或cuda环境，因此不必急于配置环境或租用算力。但之后的实验大概率会用到，如有兴趣也可一并配置。

### LR的数学原理：

模型

​	$h_\theta(x)=g(\theta^\prime x)=\frac{1}{1+exp(-\theta^\prime x)}$

训练集

​	 $\{(x^{(1)},y^{(1)}),(x^{(2)},y^{(2)}),...,(x^{(m)},y^{(m)})\}$

其中，$x=(x_0,x_1,...,x_n)'_{n+1}\in R^{n+1},x_0=1$,

$y\in \{0,1\}$,

$\theta=(\theta_0,\theta_1,...,\theta_n)'_{n+1}\in R^{n+1}$



$g(\cdot)=\frac{1}{1+exp(-\cdot)}$称为sigmoid函数，用来区分样本的类别。



损失函数: $J(\theta)=\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}log(h_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]$

损失函数的梯度$\nabla J(\theta)=\frac{1}{m}X'(g(X\theta)-Y)$

其中，

$X=\begin{pmatrix}x^{(1)'}\\x^{(2)'}\\\vdots\\x^{(m)'}\end{pmatrix}=\begin{pmatrix}1&x^{(1)'}_1&\cdots&x^{(1)'}_n\\1&x^{(2)'}_1&\cdots&x^{(2)'}_n\\\vdots&\vdots&\ddots&\vdots\\1&x^{(m)'}_1&\cdots&x^{(m)'}_n\end{pmatrix}\in R^{m\times(n+1)}$

此矩阵称为design matrix。



对于一些线性可分性较差的数据，可以引入正则项来防止其决策面过拟合。

例如，加入$L_2$正则化的损失函数:  $J(\theta)=\frac{1}{m}\sum_{i=1}^{m}[y^{(i)}log(h_\theta(x^{(i)})+(1-y^{(i)})log(1-h_\theta(x^{(i)}))]+\frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2$

### 非线性二分类：

构造高维特征: 原特征$x\in R$，可以由原特征构造特征组$(1,x,x^2)$，$(1,x,x^2,x^3)$或$(1,x,\sqrt{x})$等等。决策边界(decision boundary)也就由直线/平面变为了曲线/曲面。

以线性回归为例，原模型$h_\theta(x)=\theta_0+\theta_1 x$变为$h_\theta(x)=\theta_0+\theta_1 x+\theta_2x^2$，

$h_\theta(x)=\theta_0+\theta_1 x+\theta_2x^2+\theta_3x^3$

或$h_\theta(x)=\theta_0+\theta_1 x+\theta_2\sqrt{x}$

### 多分类原理：

多分类的原理与二分类非常类似，但不再使用sigmoid函数，因为sigmoid函数的特性决定了它只能将样本分为正负两类。

softmax函数可以解决此问题：

$\sigma(\mathbf{z})_{j}=\frac{e^{z_{j}}}{\sum_{k=1}^{K} e^{z_{k}}} \quad \text { for } j=1, \ldots, K$

其中，$e^{z_i}$为样本属于$i$类时模型的logit。此函数可以给出决策面认为样本属于各个类别的概率，从其中选择最大的概率，即为分类模型给出的样本类别。



如对本文档内容有疑问，请咨询24S103323@stu.hit.edu.cn
