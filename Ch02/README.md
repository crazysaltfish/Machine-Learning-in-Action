# 使用kNN初识机器学习
 
------
 
The code for the examples in Ch.1 is contained in the python module: kNN.py.
The examples assume that datingTestSet.txt is in the current working directory.  
Folders testDigits, and trainingDigits are assumed to be in this folder also.  
 
机器学习算法k-邻近算法（kNN），它的工作原理是：存在样本数据集，在进行分类时，将新数据的每个特征与样本集中数据对应特征进行比较，然后提取出样本集中最相似的数据（最邻近），一般只选取前k个最相似的数据。一般步骤如下：
 
> * 计算与已知类别数据集中的点与当前点之间的距离
> * 按照距离递增次序排序
> * 选取与当当前点距离最小的k个点
> * 确定前k个点所在类别的频率
> * 返回前k个点中出现频率最高的类别作为当前点的预测类别
 
以下程序基于python3.x
 
<!-- ### 1. 制作一份待办事宜 [Todo 列表](https://www.zybuluo.com/mdeditor?url=https://www.zybuluo.com/static/editor/md-help.markdown#13-待办事宜-todo-列表) -->
### 1. 使用python导入数据
 
<!-- - [ ] 支持以 PDF 格式导出文稿
- [ ] 改进 Cmd 渲染算法，使用局部渲染技术提高渲染效率
- [x] 新增 Todo 列表功能
- [x] 修复 LaTex 公式渲染问题
- [x] 新增 LaTex 公式编号功能 -->
```python
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels
```
 
### 2. 编写kNN算法
 
这里采用欧式距离计算两点的距离。

```python
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()     
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]
```
 
### 3. 使用python处理文本数据
 
首先使用line.strip()截取掉所有的回车字符，然后使用line.split('\t')将整行数据分割为元素列表。

```python
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
```
 
### 4. 使用Matplotlib创建散点图
 
使用Matplotlib库提供的scatter函数个性化的标记散点图上的点。

```python
import matplotlib
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0], datingDataMat[:,1], s=15.0*num.array(datingLabels), c=15.0*num.array(datingLabels))
plt.show()
```
 
### 5. 数值归一化
 
消除各个特征的数值差异对结果的影响。

```python
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
```

### 6. 作为完整程序测试
 
分类器针对约会网站的测试代码。

```python
def datingClassTest():
    hoRatio = 0.50      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print( "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print( "the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)
```

### 7. 构建完整可用系统
 
```python
def classifyPerson():
	resultsList = ['not at all', 'in samll dose', 'in large dose']
	percentTime = float(input("percentage of time spent at playing video games?"))
	ffMiles = float(input("frequency flier miles earned per year?"))
	iceCream = float(input("liters of ice cream consumed per year?"))
	datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
	normMat, range, minVals = autoNorm(datingDataMat)
	inArr = array([percentTime, ffMiles, iceCream])
	classifierResult = classify0((inArr - minVals)/range, normMat, datingLabels, 3)
	print("you will probably like this person:", resultsList[classifierResult - 1])
```

### 8. 示例：手写识别系统
 
- 准备数据：将图像转化为测试向量

```python
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect
```

- 测试算法：使用k-邻近算法识别手机数字

```python
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print( "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print( "\nthe total number of errors is: %d" % errorCount)
    print( "\nthe total error rate is: %f" % (errorCount/float(mTest)))
```
 
作者 [@crazysaltfish](https://github.com/crazysaltfish)     
2020 年 08月 14日    
 