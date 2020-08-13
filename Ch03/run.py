import numpy as num
import trees
import treePlotter

### 1.Test for dicision tree
## load data
myData, labels = trees.createDataSet()
# print(myData)
# print(labels)

## clac ShannonEnt
# a = trees.calcShannonEnt(myData)
# print(a)


## split dataSet
# a = trees.splitDataSet(myData, 0, 0)
# print(a)

## choose best Feature to split
# a = trees.chooseBestFeatureToSplit(myData)
# print(a, '\n', myData) 


## built tree with dict type 
subLabels = labels[:]
tree = trees.createTree(myData, subLabels)
# print(tree)

### 2. Plot tree
# treePlotter.createPlot()

## get leafs num of tree
# b = treePlotter.getNumLeafs(tree)
# print(b)

## get depth of tree
# c = treePlotter.getTreeDepth(tree)
# print(c)

# treePlotter.createPlot(tree)
# tree['no surfacing'][2] = 'maybe'
# print(tree)

### 3. test classifier
a = trees.classify(tree, labels, [1, 1])
# print(a)

### 4. Storage classifier
# trees.storeTree(tree, 'classifierStorage.txt')
# b = trees.grabTree('classifierStorage.txt')
# print(b)

### 5. Example: Use dicision-tree as contact lenses classifier
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
lensesTree = trees.createTree(lenses, lensesLabels)
print(lensesTree)
treePlotter.createPlot(lensesTree)
