import bayes

### load dataset
listPosts, listClasses = bayes.loadDataSet()
## create Vocabulary list
myVocablist = bayes.createVocabList(listPosts)
## get words to vector
# word2vec = bayes.setOfWords2Vec(myVocablist, listPosts[0]) 
## training NaiveBayes classifier
trainMat = []
for postinDoc in listPosts:
	trainMat.append(bayes.setOfWords2Vec(myVocablist, postinDoc))
	
p0, p1, pA = bayes.trainNB0(trainMat, listClasses)

print(p0, p1, pA)
