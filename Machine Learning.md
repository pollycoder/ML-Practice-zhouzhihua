# Machine Learning

## Chapt 1 绪论

### 1. 基本术语

#### (1)数据集

<font color=red>数据集</font>(dataset)：对m个样本的描述形成的集合，记为$D$.

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

所有特征张成<font color=red>属性空间</font>(特征空间、样本空间、输入空间)

特征的数目即为属性空间的<font color=red>维数</font>(dimensionality).

$\mathbf{x_i}$又被称为<font color=red>特征向量</font>(feature vector).

#### (2)学习

##### (a) <font color=red>学习</font>(learning)：从数据中学得模型的过程，又称训练(training).

<font color=yellow>Tips：也就是说学习算法本身并不是模型，学习之后获得的实例化算法才是投入使用的模型，学习算法是用来获取最终模型的.</font>

> 训练数据：训练过程中使用的数据
>
> 训练样本：训练使用的每个样本，训练样本的集合成为训练集
>
> 标记空间(label space, 输出空间)：标记的集合，记为$\Upsilon$
>
> 样例：$(\mathbf{x_i},\upsilon_i)$，前者为示例，后者为标记，也就是监督的对象

<font color=red>模型</font>(model)：对应了数据的某种潜在规律，又称<font color=orange>假设 </font>(hypothesis)

> 实际的潜在规律称为<font color=red>真相/真实</font>(ground-truth)

###### 分类问题(classification)：预测离散值

i) 二分类(binary classification)：只有两个类别

> 正类(positive class)，反类(negative class)

ii)多分类(multi-class classification)：多个类别

###### 回归问题(regression)：预测连续值

##### (b)<font color=red>测试</font>(testing)：对学得模型进行测试

测试样本(testing sample)：被预测的样本

##### (c)监督学习和无监督学习

<font color=red>监督学习</font>(supervised learning)：训练数据有标记

> 分类、回归问题

<font color=red>无监督学习</font>(unsupervised learning)：训练数据没有标记信息

> 聚类(clustering)：将训练集中的样本分成若干组(簇)

##### (d)泛化能力

泛化能力(generalization)：学得模型适用于新样本的能力

假设样本空间样本服从某分布，获得的样本服从i.i.d

### 2.假设空间

科学推理手段：归纳(induction)<!--泛化-->、演绎(deduction)<!--特化-->

归纳学习：广义、狭义

目前多数研究广义归纳学习

狭义归纳学习又称<font color=red>概念学习</font>

#### (1)概念学习

基础：布尔概念学习

数据集：
$$
D=\{特征：描述,是/否\}
$$

### (2)假设空间：所有假设组成的空间

学习过程：对假设空间进行搜索，最后获得与训练集一致的假设

<font color=yellow>版本空间(version space)：与训练集一致的“假设空间”</font>

### 3.归纳偏好

机器学习算法在学习过程中对某种类型假设的偏好称为<font color=red>归纳偏好</font>(inductive bias).

归纳偏好的作用：(图中散点代表有限训练集，但最终需要的曲线只有一条)

![bias](https://raw.githubusercontent.com/pollycoder/ML-Practice-zhouzhihua/main/img/bias.png)

#### (1)奥卡姆剃刀(Occam's razor)原则

多个假设取**<font color=red>最简</font>**，多数情况越平滑的曲线越简单

#### (2)“没有免费的午餐(No Free Lunch, NFL)”定理

对于一个学习算法$\mathcal{L}_a$，若它在某些问题上比学习算法$\mathcal{L}_b$好，则必然存在另一些问题，在那里$\mathcal{L}_b$比$\mathcal{L}_a$好

<font color=yellow>Tip:简言之就是没有适用于所有预测场景的学习算法.</font>

## Chapter 2 模型评估与选择

### 2.1 经验误差与过拟合



















