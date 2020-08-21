import bayes

### 1. prototype - (set-of-words model)
##load dataset
listPosts, listClasses = bayes.loadDataSet()
## create Vocabulary list
myVocablist = bayes.createVocabList(listPosts)
## get words to vector
# word2vec = bayes.setOfWords2Vec(myVocablist, listPosts[0]) 
## training NaiveBayes classifier
# trainMat = []
# for postinDoc in listPosts:
# 	trainMat.append(bayes.setOfWords2Vec(myVocablist, postinDoc))
	
# p0, p1, pA = bayes.trainNB0(trainMat, listClasses)

# print(p0, p1, pA)
## test the classifier
# bayes.testingNB()

### 2. use bayes to classify emails - (bag-of-words model)
# bayes.spamTest()

### 3. use bayes to get interesets from personal ads.
import feedparser
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
# vocabList, pSF, pNY = bayes.localWords(ny, sf)
bayes.getTopWords(ny, sf)

