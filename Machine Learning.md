# Machine Learning

## Chapt 1 绪论

### 1. 基本术语

#### (1)数据集

$\color{red}{数据集(dataset)}$：对m个样本的描述形成的集合，记为$D$.

$$
\begin{align}
&D=\{\mathbf{x_1},\mathbf{x_2},...\mathbf{x_m}\} \\\\
&\mathbf{x_i}=\left[
\begin{matrix}
特征(1):描述(1)\\
特征(2):描述(2)\\
...\\
特征(j):描述(j)
\end{matrix}\right]\\\\
\end{align}
$$

所有特征张成$\color{red}{属性空间}$(特征空间、样本空间、输入空间)

特征的数目即为属性空间的<font color=red>维数</font>(dimensionality).

$\mathbf{x_i}$又被称为<$\color{red}{特征向量}$(feature vector).

#### (2)学习

##### (a) $\color{red}{学习}$(learning)：从数据中学得模型的过程，又称训练(training).

$\color{yellow}{Tips：也就是说学习算法本身并不是模型，学习之后获得的实例化算}$

$\color{yellow}{法才是投入使用的模型，学习算法是用来获取最终模型的.}$

> 训练数据：训练过程中使用的数据
>
> 训练样本：训练使用的每个样本，训练样本的集合成为训练集
>
> 标记空间(label space, 输出空间)：标记的集合，记为$\Upsilon$
>
> 样例：$(\mathbf{x_i},\upsilon_i)$，前者为示例，后者为标记，也就是监督的对象

$\color{red}{模型}$(model)：对应了数据的某种潜在规律，又称$\color{orange}{假设 }$(hypothesis)

> 实际的潜在规律称为$\color{red}{真相/真实}$(ground-truth)

###### 分类问题(classification)：预测离散值

i) 二分类(binary classification)：只有两个类别

> 正类(positive class)，反类(negative class)

ii)多分类(multi-class classification)：多个类别

###### 回归问题(regression)：预测连续值

##### (b)$\color{red}{测试}$(testing)：对学得模型进行测试

测试样本(testing sample)：被预测的样本

##### (c)监督学习和无监督学习

$\color{red}{监督学习}$(supervised learning)：训练数据有标记

> 分类、回归问题

$\color{red}{无监督学习}$(unsupervised learning)：训练数据没有标记信息

> 聚类(clustering)：将训练集中的样本分成若干组(簇)

##### (d)泛化能力

泛化能力(generalization)：学得模型适用于新样本的能力

假设样本空间样本服从某分布，获得的样本服从i.i.d

### 2.假设空间

科学推理手段：归纳(induction)<!--泛化-->、演绎(deduction)<!--特化-->

归纳学习：广义、狭义

目前多数研究广义归纳学习

狭义归纳学习又称$\color{red}{概念学习}$

#### (1)概念学习

基础：布尔概念学习

数据集：
$$
D=\{特征：描述,是/否\}
$$

### (2)假设空间：所有假设组成的空间

学习过程：对假设空间进行搜索，最后获得与训练集一致的假设

$\color{red}{版本空间(version space)}$：与训练集一致的“假设空间”

### 3.归纳偏好

机器学习算法在学习过程中对某种类型假设的偏好称为$\color{red}{归纳偏好}$(inductive bias).

归纳偏好的作用：(图中散点代表有限训练集，但最终需要的曲线只有一条)

![bias](https://raw.githubusercontent.com/pollycoder/ML-Practice-zhouzhihua/main/img/bias.png)

#### (1)奥卡姆剃刀(Occam's razor)原则

多个假设取$\color{red}{最简}$，多数情况越平滑的曲线越简单

#### (2)“没有免费的午餐(No Free Lunch, NFL)”定理

对于一个学习算法$\mathcal{L}_a$，若它在某些问题上比学习算法$\mathcal{L}_b$好，则必然存在另一些问题，在那里$\mathcal{L}_b$比$\mathcal{L}_a$好

$\color{yellow}{Tip:简言之就是没有适用于所有预测场景的学习算法.}$

## Chapter 2 模型评估与选择

### 2.1 经验误差与过拟合

误差(error)：学习器实际预测输出和样本真实输出之间的差异

经验误差(empirical error)：学习器在训练集上的误差

主要问题：过拟合

$\color{red}{Def~~学习算法的学习能力过强以至于把学习样本}$

$\color{red}{的特殊规律当做一般规律学习称为过拟合.}$

模型选择：算法和参数的选择

### 2.2 评估方法

主流方法：实验测试，以测试集的测试误差近似泛化误差

#### (1) 从数据集中生成训练集和测试集

有一包含$m$个样例的数据集$D=\{(\mathbf{x_1},y_1),(\mathbf{x_2},y_2),...,(\mathbf{x_m},y_m)\}$，现要从中产生出训练集$S$和测试集$T$.

##### (a) 留出法

方法：直接划成互斥的两个集合

原则：若干次随机划分，重复实验评估后取平均，一般取2/3~4/5用于训练.

##### (b) 交叉验证法

方法：划分成k个大小相似的互斥子集，一个用于测试，其他用于训练

优缺点：准确，但开销大

##### (c)自助法(bootstrap)

方法：产生采样数据集D'：每次随机从D中选取一个样本，拷贝后放入D'，再放回D，重复执行m次.

多用于数据集小，难以有效划分时，能从初始数据集产生多个不同数据集

但改变了初始数据集分布，从而引入估计偏差

#### (2)调参与最终模型

对每个参数选定一个范围和变化步长，从中选出一个选定值

学的模型在实际使用中遇到的数据：测试数据

模型评估与选择用于评估测试的数据集：验证集

### 2.3 性能度量

回归任务：均方误差
$$
E(f;D)=\frac{1}{m}\sum_{i=1}^{m}(f(x_i)-y_i)^2.\\\\
E(f;D)=\int_{x-D}(f(\mathbf{x})-y)^2p(\mathbf{x})d\mathbf{x}
$$
分类任务：

#### (1) 错误率&精度

Error:
$$
E(f;D)=\frac{1}{m}\sum_{i=1}^{m}II(f(x_i)\neq y_i)\\
E(f;D)=\int_{x-D}II(f(x_i)\neq y_i)p(\mathbf{x})d\mathbf{x}\\
$$
Accuracy:
$$
Acc(f;D)=\frac{1}{m}\sum_{i=1}^{m}II(f(x_i)= y_i)\\
Acc(f;D)=\int_{x-D}II(f(x_i)= y_i)p(\mathbf{x})d\mathbf{x}\\
$$

#### (2)查准率、查全率、F1

查准率(precision)：真阳率   $\to 1$代表找到的阳性都是真的，但不一定所有阳性都被找到

查全率(recall)：阳性中预测为阳性的比例 $\to 1$代表所有阳性都被找到，但找到的目标中有假阳性

$\textcolor{yellow}{Tip:~查准和查全通常不能兼顾，视需求而定.}$

P-R曲线：precision-recall，依据学习器的预测结果对样例排序，recall为横轴（意为从最可能正的一直取，取到所有正例都被取出为止）

性能度量：P-R曲线下包裹的面积、平衡点(BEP)

对查准/查全率的偏好定义：
$$
F_{\beta}=\frac{(1+\beta^2)\cross P\cross R}{(\beta^2\cross P)+R}
$$
F1: $\beta=1$时。$\beta$表示查全率对查准率的重要性
$$
\beta>1:查全率影响大\\
\beta<1:查准率影响大
$$
n个二分类混淆矩阵中考察查准率和查全率:

(a)宏查准率，宏查全率：先算查准（全）率，再取平均
$$
macro-P=\frac{1}{n}\sum_{i=1}^{n}P_i\\
macro-R=\frac{1}{n}\sum_{i=1}^{n}R_i\\
macro-F1=\frac{2\cross macro-P\cross macro-R}{macro-P+macro-R}
$$
(b)微查准率，微查全率：先把混淆矩阵混合，再对TP,FP,TN,FN求平均
$$
micro-P=\frac{\bar{TP}}{\bar{TP}+\bar{FP}}\\
micro-R=\frac{\bar{TP}}{\bar{TP}+\bar{FN}}\\
micro-F1=\frac{2\cross micro-P\cross micro-R}{micro-P+micro-R}
$$

#### (3)ROC,AUC

分类阈值(threshold)，大于为正，小于为负

![Screenshot 2023-02-02 at 21.33.12](/Users/polly/Library/Application Support/typora-user-images/Screenshot 2023-02-02 at 21.33.12.png)

##### 绘制方法：

给定$m^+,m^-$个正例和负例，根据学习器预测结果排序，将分类阈值调到最大，然后将阈值依次设为每个样例的预测值，

##### 损失(loss)：

$$
\begin{align}
l_{rank}=&\frac{1}{m^+m^-}(II(f(\mathbf{x^+})<f(\mathbf{x^-}))\\&+\frac{1}{2}II(f(\mathbf{x^+})=f(\mathbf{x^-})))
\end{align}
$$



> 解释：考虑每一对正负例，
>
> 若正例预测值小于反例(<font color=yellow>完全错了</font>)，则罚分为1；
>
> 若正例预测值等于反例(<font color=yellow>错了一半</font>)，则罚分为0.5.
>
> 以上图为例，如果正确识别，则应该假正例率不变，真正例率增加$\frac{1}{m_+}$,对应下图的绿线
>
> 如果错误识别，则应该真正例率不变，假正例率增加$\frac{1}{m_-}$，对应下图的红线
>
> 如果同时有正确和错误识别，则对应蓝线

![Screenshot 2023-02-08 at 11.22.19](/Users/polly/Library/Application Support/typora-user-images/Screenshot 2023-02-08 at 11.22.19.png)
$$
AUC=1-l_{rank}
$$

#### (4)代价敏感错误率与代价曲线

<font color=red>最终目标：最小化总体代价(total cost)</font>

非均等代价(unequal cost)：权衡不同错误的后果

代价矩阵：正确预测均无代价，损失对比看比值

![Screenshot 2023-02-08 at 11.33.22](/Users/polly/Library/Application Support/typora-user-images/Screenshot 2023-02-08 at 11.33.22.png)

代价敏感错误率：
$$
E(f;D;cost)=\frac{1}{m}(\sum_{x_i\in D^+}II(f(x_i\neq y_i)\cross cost_{01}\\+\sum_{x_i\in D^-}II(f(x_i\neq y_i)\cross cost_{10}
$$
可在此基础上给出基于分布的敏感错误率

代价曲线：

![Screenshot 2023-02-08 at 13.23.32](/Users/polly/Library/Application Support/typora-user-images/Screenshot 2023-02-08 at 13.23.32.png)

正例概率代价：
$$
P(+)_{cost}=\frac{p\cross cost_{01}}{p\cross cost_{01}+(1-p)\cross cost_{10}}
$$

> p为样例为正例的概率
>
> 解释：
>
> 分母代表样例本身是正例且预测错误的加权代价；
>
> 分子代表所有预测错误的情形的代价总和；
>
> 也就是说原式代表已知预测错误的情况下样例实际为正例的相对代价

归一化代价：
$$
cost_{norm}=\frac{FNR\cross p\cross cost_{01}+FPR\cross (1-p)\cross cost_{10}}{p\cross cost_{01}+(1-p)\cross cost_{10}}
$$

>FPR:假正例率
>
>FNR:假负例率

曲线绘制：

ROC上每一点对应代价平面上端点为(0,FPR)和(1,FNR)的线段，把所有线画出来下面的公共区域面积即为所有条件下学习器的期望总体代价

### 2.4 比较检验

#### (1)假设检验

假设：对学习器泛化错误率分布的判断

本质：在假设成立的条件下，反常的实验应当是小概率事件，所以当实验现象出现在概率小于我们设定的显著度$\alpha$的区域（拒绝域）时，就拒绝假设.

> e.g. $H_0$: $\epsilon=\epsilon_0$
>
> e.g.$H_0$: $\epsilon\leq\epsilon_0$

##### 1)二项检验

<font color=red>假设：单个模型的泛化错误率 </font>$\textcolor{red}{\epsilon\leq\epsilon_0}$，符合二项分布
$$
\bar{\epsilon}=max_{\epsilon}(s.t.\sum_{i=\epsilon_0\cross m+1}^{m}(^m_i)\epsilon^{i}(1-\epsilon)^{m-i}<\alpha)
$$

##### 2)t-检验

<font color=red>假设：k个模型泛化错误率平均值为 </font>$\textcolor{red}{\mu}$<font color=red>，假设</font>$\textcolor{red}{\epsilon=\mu.}$

错误率：
$$
\mu=\frac{1}{k}\sum_{i=1}^{k}\hat{\epsilon_i}
$$
方差
$$
\sigma^2=\frac{1}{k-1}\sum_{i=1}^{k}(\hat{\epsilon_i}-\mu)^2
$$
统计变量
$$
\tau_t =\frac{\sqrt{k}(\mu-\epsilon_0)}{\sigma}
$$
符合自由度为k-1的t分布

##### 3)交叉验证t检验

#####  
