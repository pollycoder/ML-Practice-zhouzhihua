# Practice of Machine Learning

## Chapter 1 (感谢知乎版主wottzh)

### 1.1

![Screenshot 2023-01-19 at 17.38.42](/Users/polly/Library/Application Support/typora-user-images/Screenshot 2023-01-19 at 17.38.42.png)

### 1.2

![Screenshot 2023-01-19 at 19.42.10](/Users/polly/Library/Application Support/typora-user-images/Screenshot 2023-01-19 at 19.42.10.png)

不包括\*，一共有2\*3\*3=18种单个合取式表示的假设

因而假设数上限为$2^{18}$.

如果考虑\*，则在48个特征组合中随机选取k个组合，其中要注意避免\*导致的假设空间重复.

从可能带星号的析合范式获取实际假设空间：

(1)生成一三维向量a，第一维表示色泽，第二维表示根蒂，第三维表示敲声，带星号则记为2.

(2)将a对应成18维向量，即为A的实际假设

规则：前9个表示A\[0]\[:]\[:]，即青绿色；后9个表示A\[1]\[:]\[:]，即乌黑色；

每个“9人组”前3个表示A\[:]\[0]\[:]，即根蒂蜷缩，以此类推...

如果带星号，就将该属性对应的3个位置全部置为1

(3)把48种矩阵全部生成18维向量，然后遍历选取k个，再空出一个18维0向量记录，只要k个向量有一个该位置是1，则记为1，得到最终序列，生成一个含有(待计算)个列向量的列表

(4)使用set去重，再计算长度即为个数

源代码见`ML_practice_src/1.2.py`

### 1.3

在训练过程中选择满足最多样本的假设。也可以对每个假设，求得其准确率。准确率=(符合假设的条件且为好瓜的样例数量)/(符合假设的条件的样例数量)。选择准确率最高的假设。

### 1.4

![Screenshot 2023-01-20 at 11.51.30](/Users/polly/Library/Application Support/typora-user-images/Screenshot 2023-01-20 at 11.51.30.png)



