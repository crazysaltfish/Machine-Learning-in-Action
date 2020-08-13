import numpy as num
import kNN 
import matplotlib
import matplotlib.pyplot as plt

## kNN test function
group,labels = kNN.createDataSet()
result = kNN.classify0([0,0], group, labels, 3)

### 1.yuehui wangzhan peidui
## load data and dating
datingDataMat, datingLabels = kNN.file2matrix('datingTestSet2.txt')

## plot dataSet 
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(datingDataMat[:,0], datingDataMat[:,1], s=15.0*num.array(datingLabels), c=15.0*num.array(datingLabels))
# plt.show()

## normalization
normMat, valueRange, minVals = kNN.autoNorm(datingDataMat)

## test the model
# kNN.datingClassTest()

### 2.shouxie shibei xitong
kNN.handwritingClassTest()

## a complete classifier system 
# kNN.classifyPerson()




# print(normMat)
# print(range)
# print(minVals)
