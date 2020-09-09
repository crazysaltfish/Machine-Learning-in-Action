import numpy as num 
import logRegres

dataMat, labelMat = logRegres.loadDataSet()
# weights = logRegres.gradAscent(dataMat, labelMat)

## Draw decision boundaries
# 将wights转化为数组
# logRegres.plotBestFit(weights.getA())

## Random gradascent
# weights = logRegres.stocGradAscent0(num.array(dataMat), labelMat)
# logRegres.plotBestFit(weights)

## Improved gradascent 
# weights = logRegres.stocGradAscent1(num.array(dataMat), labelMat)
# logRegres.plotBestFit(weights)

## Example classify the death rate of horse
logRegres.multiTest()