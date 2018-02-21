import os
import sys
import json
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from collections import Counter
import statsmodels.api as sm
import statsmodels.formula.api as smapi
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from gensim import corpora, models
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import Embedding

def writeScore(dictFile, fileName):
	theFile = open(fileName, "w")
	for item in dictFile:
		theFile.write("%s: " % str(item))
		theFile.write("%s\n" % str(dictFile[item]))
	theFile.close()
	return

def writeList(listFile, fileName):
	theFile = open(fileName, "w")
	for item in listFile:
		theFile.write("%s\n" % str(item))
	theFile.close()
	return

def writeWordEmbLists(wordembsamplesArray, wordembsamplesArrayPredict, fileName):
	tf = open(fileName, "w")
	years = len(wordembsamplesArray)
	phrases = len(wordembsamplesArray[0])
	totalSpaces = 35
	finalYear = 2014
	startingYear = finalYear - years
	for yearIndex in range(years):
		tf.write("%s Actual -------------------- Predict\n" % str(startingYear + yearIndex))
		for phraseIndex in range(phrases):
			actualPhrase = wordembsamplesArray[yearIndex][phraseIndex]
			if actualPhrase in wordembsamplesArrayPredict[yearIndex]:
				actualPhrase += "\t++"
			remainingSpaces = totalSpaces - len(actualPhrase)
			tf.write("%s" % actualPhrase)
			
			for n in range(remainingSpaces):
				tf.write(" ")
			predictPhrase = wordembsamplesArrayPredict[yearIndex][phraseIndex]
			if predictPhrase in wordembsamplesArray[yearIndex]:
				predictPhrase += "\t++"
			tf.write("%s\n" % predictPhrase)

		tf.write("\n\n")
	tf.close()
	return

def writeTopPhrasesList(topPhrases, baseYear, fileName):
	tf = open(fileName, "w")
	index = 0
	for yearPhrase in topPhrases:
		tf.write("%s: \n" % str(index + baseYear))
		for topic in yearPhrase:
			for phrase in topic:
				tf.write("%s\n" % str(phrase.encode('utf-8')))
			tf.write("\n")

		tf.write("\n")


		index += 1

	tf.close()
	return

def readVocabSeries(fileName):
	fp = open(fileName, 'r')
	vocabList = []
	for line in fp:
		vocab = line.strip("\n").split(", ")
		vocabList.append(vocab)

	fp.close()
	return vocabList

def readTimeSeriesData(fileName):
	fp = open(fileName, "r")
	phraseList = []
	timeSeries = []
	for line in fp:
		temp = line.replace("nan ", "0 ").split(":")
		phraseList.append(temp[0])
		timeSeries.append([float(ele) for ele in temp[1].split(" ")])

	return phraseList, timeSeries

def splitData2(timeSeries, windowSize):
	resultXList = []
	resultYList = []

	tempLine = timeSeries[0]
	size = len(tempLine) - windowSize
	# print size
	seriesSize = len(timeSeries)
	# print seriesSize
	for index in range(size):
		resultX = []
		resultY = []
		for lineIndex in range(seriesSize):
			training = timeSeries[lineIndex][index: index + windowSize]

			# temp = training[-1]
			# training.append(temp ** 2)

			# for windowStep in range(windowSize):
			# 	training.append(training[windowStep] ** 2)
			# for intermediate in range(windowSize - 1):
			# 	training.append(training[intermediate + 1] - training[intermediate])
			training = addFeatures(training, 5)
			training.insert(0, lineIndex)
			resultX.append(training)
			resultY.append(timeSeries[lineIndex][index + windowSize])
		resultXList.append(resultX)
		resultYList.append(resultY)

	return resultXList, resultYList, size

def splitData3(timeSeries, windowSize):
	resultX = []
	resultY = []
	powerTerms = []

	tempLine = timeSeries[0]
	size = len(tempLine) - windowSize

	seriesSize = len(timeSeries)

	for index in range(size):
		for lineIndex in range(seriesSize):
			training = timeSeries[lineIndex][index: index + windowSize]

			# training = addFeaturesKF(training, 4)
			# powerTerms.append(combination(training))
			
			training.insert(0, lineIndex) # format of data: [year index, phrase index, scores]
			training.insert(0, index)
			resultX.append(training)
			resultY.append(timeSeries[lineIndex][index + windowSize])
			
	return resultX, resultY, powerTerms, size

def combination(currentTrainingSample):
	''' The power set iterates through all the possible combination of additional features '''
	powerset = []
	windowSize = len(currentTrainingSample)
	for i in range(windowSize):
		for j in range(i, windowSize):
			powerset.append(currentTrainingSample[i] * currentTrainingSample[j])
	return powerset

def generatePowerset(s):
	resultList = []
	x = len(s)
	for i in range(1<<x):
		resultList.append([s[j] for j in range(x) if (i & (1 << j))])

	return resultList

def mapPhraseListUsingIndex(phraseIndexList, phraseList):
	# result = [phraseList[ele] for ele in phraseIndexList]
	result = []
	for index in range(len(phraseList)):
		if index in phraseIndexList:
			result.append(phraseList[index])
	return result

def writePhraseListTotal(phraseList, fileName):
	tf = open(fileName, "w")

	for phraseListThisYear in phraseList:
		for phraseListSub in phraseListThisYear:
			tf.write("%s\n" % str(phraseListSub))
		tf.write("\n")

	tf.close()
	return

def isSubarray(small, big):
	windowSize = len(small)
	steps = len(big) - windowSize + 1
	for startingIndex in range(steps):
		testArray = big[startingIndex:startingIndex + windowSize]
		if np.array_equal(testArray, small):
			return True
	return False

def checkPrecisionRecall(Xdata, Ydata, Yprediction):
	Xdata = np.asarray(Xdata)
	Ydata = np.asarray(Ydata)
	Yprediction = np.asarray(Yprediction)
	maxXdata = np.amax(Xdata, axis = 1)
	actualDist = maxXdata < Ydata
	predictDist = maxXdata < Yprediction

	TP = np.count_nonzero(actualDist * predictDist)
	TN = np.count_nonzero(actualDist) - TP
	precision, recall, f1 = 0, 0, 0
	if np.count_nonzero(predictDist) != 0:
		precision = np.float(TP) / np.count_nonzero(predictDist)
	else:
		precision = 'NA'

	if np.count_nonzero(actualDist) != 0:
		recall = np.float(TP) / np.count_nonzero(actualDist)
	else:
		recall = 'NA'

	if precision != 'NA' and recall != 'NA' and (precision + recall) > 0:
		f1 = 2 * (precision * recall) / (precision + recall)
	return precision, recall, f1

def retrieveTrendingIndices(Xdata, Ydata, Yprediction):
	Xdata = np.asarray(Xdata)
	Ydata = np.asarray(Ydata)
	Yprediction = np.asarray(Yprediction)
	maxXdata = np.amax(Xdata, axis = 1)
	actualDist = maxXdata < Ydata
	predictDist = maxXdata < Yprediction

	actualDist = Ydata - maxXdata
	predictDist = Yprediction - maxXdata

	def filterNonZero(x): 
		if x <= 0:
			return 0
		return x
	actualDist = np.asarray([filterNonZero(x) for x in actualDist])
	predictDist = np.asarray([filterNonZero(x) for x in predictDist])

	TPdist = np.asarray(actualDist * predictDist)

	actualDistIndices = np.argpartition(actualDist, -20)[-20:]
	predictDistIndices = np.argpartition(predictDist, -20)[-20:]

	TPDistIndices = np.argpartition(TPdist, -20)[-20:]

	return actualDistIndices, predictDistIndices, TPDistIndices

def calcMRRMAPNDCG(actualIndices, predictIndices):
	# predictIndices = actualIndices[:]
	# print actualIndices
	# print predictIndices
	scores = np.asarray([float(1) / (i + 1) for i in range(len(actualIndices))])
	predictScores = np.asarray([0 for n in range(len(actualIndices))],dtype=float)
	num = len(actualIndices)
	DCG_GT = scores[0]

	for index in range(1, num):
		DCG_GT += (scores[index] / math.log((index + 1), 2))

	mask = actualIndices == predictIndices
	predictScores[mask] = scores[mask]

	DCG_Pred = predictScores[0]
	for index in range(1, num):
		DCG_Pred += (predictScores[index] / math.log((index + 1), 2))

	nDCG = DCG_Pred / DCG_GT

	return nDCG

def scale(originalData):
	npData = np.array(originalData)
	currMin = npData.min()
	currMax = npData.max()
	result = (npData - currMin) / (currMax - currMin)
	return result

def scale2DArray(originalData):
	npData = np.array(originalData)
	result = []
	for currRow in npData:
		currMin = currRow.min()
		currMax = currRow.max()
		result.append((currRow - currMin) / (currMax - currMin))
	result = np.array(result)
	return result

def formWordEmbeddingTrainingData(seqOfPhrases, emb, embvocab):
	windowSize = 3
	decayingWeights = [0.8 ** n for n in range(windowSize)][::-1]
	predefinedK = 20
	totalYear = len(seqOfPhrases)
	resultX = []
	resultY = []
	fullResultX = []
	fullResultY = []
	for yearIndex in range(totalYear - windowSize):
		print "Currently processing year: " + str(yearIndex)
		yearWords = [seqOfPhrases[yearIndex + n] for n in range(windowSize + 1)]

		# year1Words = seqOfPhrases[yearIndex]
		# year2Words = seqOfPhrases[yearIndex + 1]
		# year3Words = seqOfPhrases[yearIndex + 2]
		# year4Words = seqOfPhrases[yearIndex + 3]
		for phraseIndex in range(predefinedK):
			print "getting phrases: " + str(phraseIndex)
			currYearWords = [yearWords[n][:(phraseIndex + 1)] for n in range(windowSize + 1)]
			# currYear1Words = year1Words[:(phraseIndex + 1)]
			# currYear2Words = year2Words[:(phraseIndex + 1)]
			# currYear3Words = year3Words[:(phraseIndex + 1)]
			# currYear4Words = year4Words[:(phraseIndex + 1)]
			currYearMatrices = [[] for n in range(windowSize + 1)]
			currYearMatrices = map(lambda x: np.asarray([emb.wv[ele] for ele in x if ele in embvocab]), currYearWords)
			# for currYearWordList in currYearWords:
			# 	for currYearWord in currYearWordList:
			# 		if currYearWord in embvocab:
			# 			currYearVector = emb.wv[currYearWord]
			currYearVectors = [np.average(innerMatrix, axis = 0) for innerMatrix in currYearMatrices]
			trainingY = currYearVectors.pop()
			trainingX = np.average(np.asarray(currYearVectors), axis = 0, weights = decayingWeights)

			resultX.append(trainingX)
			resultY.append(trainingY)

			if phraseIndex == (predefinedK - 1):
				fullResultX.append(trainingX)
				fullResultY.append(trainingY)

	return np.asarray(resultX), np.asarray(resultY), np.asarray(fullResultX), np.asarray(fullResultY)

def trainEmbWords(trainX, trainY, fullX):
	model = Sequential()
	lookback = len(trainX[0])
	trainXLSTM = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	trainFullX = np.reshape(fullX, (fullX.shape[0], 1, fullX.shape[1]))
	print "Training LSTM for word embeddings"
	model.add(LSTM(200, return_sequences = True, input_shape = trainXLSTM.shape[1:]))
	# model.add(Dropout(0.5))
	model.add(LSTM(200, return_sequences = True))
	# model.add(Dropout(0.5))
	model.add(LSTM(200))
	# model.add(Dropout(0.5))
	model.add(Dense(lookback))
	model.compile(loss="mean_squared_error", optimizer = "rmsprop")
	model.fit(trainXLSTM, trainY, epochs = 500, batch_size = 10, verbose = 2)

	fullYPredict = model.predict(trainFullX)


	# lookback = len(newXTrain[0])
	# XTrainLSTM = np.reshape(newXTrain, (newXTrain.shape[0], newXTrain.shape[1], 1))
	# XTestLSTM = np.reshape(newXTest, (newXTest.shape[0], newXTest.shape[1], 1))
	# LSTMModel = Sequential()

	# LSTMModel.add(SimpleRNN(4, return_sequences = True, input_shape = XTrainLSTM.shape[1:]))
	# LSTMModel.add(SimpleRNN(4))
	# LSTMModel.add(Dense(1))
	# LSTMModel.compile(loss="mean_squared_error", optimizer = 'rmsprop')
	# LSTMModel.fit(XTrainLSTM, newYTrain, epochs = 100, batch_size = 10, verbose = 2)

	# LSTMTrainPredict = LSTMModel.predict(XTrainLSTM)
	# LSTMTestPredict = LSTMModel.predict(XTestLSTM)

	return fullYPredict

def prepareWordWithGoogleMatrix(wordList, timeSeries, baseYear, vocabTfIdfSeriesMap):
	yearCover = len(timeSeries[0])
	phraseCount = len(wordList)
	result = [[0.0 for n in range(yearCover)] for i in range(phraseCount)]
	model = models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
	# vocabSeries = readVocabSeries("single/vocabSeries.txt")

	vectorShape = model.wv['the'].shape

	vectors = []
	topPhrases = []
	timeSeries = np.asarray(timeSeries)
	
	wordListCopy = list(wordList)
	wordList = np.asarray(wordList)

	for yearIndex in range(yearCover):
		predefinedNumOfClusters = 25
		# vocab = vocabSeries[index]
		currVocab = wordList[np.nonzero(timeSeries[:, yearIndex])[0]]
		vectors = []

		missingList = []
		for NP in currVocab:
			try:
				words = NP.split(" ")
				tfIdfSum = 0.0
				vectorSum = np.zeros(vectorShape) # initialize the zero vector
				for individualWord in words:
					currVector = model.wv[individualWord]
					currTfIdf = vocabTfIdfSeriesMap[individualWord][yearIndex]
					# print individualWord
					# print currTfIdf
					tfIdfSum += currTfIdf
					vectorSum = vectorSum + currVector * currTfIdf

					# print vectorSum

				vectors.append(vectorSum / tfIdfSum)
			except KeyError, e:
				print "The error is: %s, in year %d" % (e, yearIndex + baseYear)
				missingList.append(NP)
				pass


		currVocab = currVocab.tolist()
		for missingNP in missingList:
			currVocab.remove(missingNP)

		print "Now running K Means"
		numOfClusters = predefinedNumOfClusters if len(vectors) > predefinedNumOfClusters else len(vectors)

		# print vectors
		print np.array(vectors)
		try:
			kmeans = KMeans(n_clusters = numOfClusters, random_state = 0).fit(np.array(vectors))
		except ValueError, e:
			print "The array problem is: %s, in year %d" % (e, yearIndex + baseYear)
			continue
		print "Finished running K Means"
		centers = kmeans.cluster_centers_
		smallestDistance = 100.0
		for phraseIndex in range(len(currVocab)):
			word = currVocab[phraseIndex]
			vector = vectors[phraseIndex]
			label = kmeans.labels_[phraseIndex]
			centroid = centers[label]
			distance = np.linalg.norm(vector - centroid)
			relativeness = cossim(vector, centroid)
			if word in wordListCopy:
				# newListIndex = np.where(wordList == word)[0]
				newListIndex = wordListCopy.index(word)
				result[newListIndex][yearIndex] = relativeness

		currTopPhrases = []
		for center in centers:
			topKeyphrases = [ele[0] for ele in model.similar_by_vector(center, topn = 10)]
			currTopPhrases.append(topKeyphrases)
		topPhrases.append(currTopPhrases)

	return result, topPhrases

def preparePhraseRelativenessMatrix(phraseList, timeSeries, baseYear):
	yearCover = len(timeSeries[0])
	phraseCount = len(phraseList)
	result = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	embFileList = os.listdir('single/emblistflat/')
	continuousEmbFileList = os.listdir('single/emblist/')
	topPhrases = []
	vocabMap = {}
	for fileName in embFileList:

		year = int(fileName.split("-")[1])

		print "processing file iter 1: %s" % str(year)

		model = models.Word2Vec.load('single/emblistflat/' + fileName)
		vocab = list(model.wv.vocab)
		vocabMap[year] = vocab

	for fileName in continuousEmbFileList:

		year = int(fileName.split("-")[1])

		print "processing file iter 2: %s" % str(year)

		model = models.Word2Vec.load('single/emblist/' + fileName)
		vocab = vocabMap[year]
		predefinedNumOfClusters = 15
		numOfClusters = int(math.sqrt(len(vocab)))
		numOfClusters = predefinedNumOfClusters if predefinedNumOfClusters < len(vocab) else len(vocab)
		vectors = []
		for word in vocab:
			vectors.append(model.wv[word])

		print "Now running K Means"
		kmeans = KMeans(n_clusters = numOfClusters, random_state = 0).fit(np.array(vectors))
		print "Finished running K Means"
		centers = kmeans.cluster_centers_
		smallestDistance = 100.0
		for phraseIndex in range(len(vocab)):
			word = vocab[phraseIndex]
			vector = vectors[phraseIndex]
			label = kmeans.labels_[phraseIndex]
			centroid = centers[label]
			distance = np.linalg.norm(vector - centroid)
			if distance == 0.0:
				distance = smallestDistance / 100    # think about another way to do this, otherwise is magic number
			elif distance < smallestDistance:
				smallestDistance = distance
			relativeness = 1.0 / distance
			relativeness = cossim(vector, centroid)
			if word in phraseList:
				newListIndex = phraseList.index(word)
				result[newListIndex][year - baseYear] = relativeness

		currTopPhrases = []
		for center in centers:
			topKeyphrases = [ele[0] for ele in model.similar_by_vector(center, topn = 10)]
			currTopPhrases.append(topKeyphrases)
		topPhrases.append(currTopPhrases)


	return result, topPhrases

def normalize2DArrayVertical(originalArray):
	numOfColumns = len(originalArray[0])
	for index in range(numOfColumns):
		oldArray = originalArray[:, index]
		newArray = (oldArray-float(min(oldArray)))/(max(oldArray)-min(oldArray))
		originalArray[:, index] = newArray
	return originalArray

def normalize2DArrayHorizontal(originalArray):
	newArray = []
	for vector in originalArray:
		vector = np.asarray(vector)
		norm = np.linalg.norm(vector)
		if norm == 0:
			newArray.append(vector.tolist())
		newArray.append((vector / norm).tolist())
	return newArray

def normLinear(originalArray):
	numOfRows = len(originalArray)
	for index in range(numOfRows):
		row = np.array(originalArray[index])
		print row
		newArray = (row - float(min(row))) / (max(row) - min(row))
		print newArray
		originalArray[index] = newArray.tolist()
	return originalArray

def cossim(a, b):    # a and b are 1D numpy array and should have the same dim
	dot = np.dot(a, b)
	a_norm = np.linalg.norm(a)
	b_norm = np.linalg.norm(b)
	return dot / (a_norm * b_norm)

if (__name__ == "__main__"):
	# fileName = "json/soft-abs-nolast-ori.txt"
	# fileName = "single/soft.txt"
	# fileName = "np-doublegraph/doublegraph-5000-NPs.txt"
	vocabFileName = "np-doublegraph/doublegraph-NPs-vocab.txt"
	fileName = "np-doublegraph/TextRankbaseline.txt"
	aggregateCoefficient = 0.2
	phraseList, timeSeries = readTimeSeriesData(fileName)
	vocabList, tfIdfSeries = readTimeSeriesData(vocabFileName)
	vocabTfIdfSeriesMap = {}
	for index in range(len(vocabList)):
		vocabTfIdfSeriesMap[vocabList[index]] = tfIdfSeries[index]

	windowSize = 3
	testSize = 20
	numOfTopics = 10
	checkAllowance = 0

	baseYear = 1954

	newXList, newYList, powerTerms, yearCover = splitData3(timeSeries, windowSize)
	newXList = np.asarray(newXList)
	newYList = np.asarray(newYList)

	print "finished splitting data"

	regression = linear_model.LinearRegression()
	phraseCount = len(phraseList)
	# meanSquareError = {}
	# varianceScore = {}
	# coefRegression = {}

	# RMSErrorTermListMap = {phrase:[0 for n in range(yearCover)] for phrase in phraseList}
	# RMSErrorTermList = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	RMSErrorTerm = []
	DIFFTermListActual = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	DIFFTermListPrediction = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	precisionRecallList = [(0, 0, 0) for n in range(yearCover)]


	newXTrainWithIndices, newXTestWithIndices, newYTrain, newYTest = train_test_split(newXList, newYList, test_size = 0.2, random_state = 42)

	newXTrain = np.delete(newXTrainWithIndices, np.s_[0:2], axis = 1) # format of data: [year index, phrase index, scores]
	newXTest = np.delete(newXTestWithIndices, np.s_[0:2], axis = 1)

	newXTrainYear = newXTrainWithIndices[:, 0].astype(int)
	newXTestYear = newXTestWithIndices[:, 0].astype(int)
	newXTrainIndices = newXTrainWithIndices[:, 1].astype(int)
	newXTestIndices = newXTestWithIndices[:, 1].astype(int)

	newXTrainBkp = np.copy(newXTrain)
	newYTrainBkp = np.copy(newYTrain)

	newXListWithoutIndices = np.delete(newXList, np.s_[0:2], axis = 1)

	# print "length of training"
	# print len(newXTrain)
	# print "before statsmodel"


	"""
	This is where the LSTM comes in;
	the current implementation assumes single feature LSTM and all 3 time steps are consolidated into one feature

	"""
	# lookback = len(newXTrain[0])
	# XTrainLSTM = np.reshape(newXTrain, (newXTrain.shape[0], newXTrain.shape[1], 1))
	# XTestLSTM = np.reshape(newXTest, (newXTest.shape[0], newXTest.shape[1], 1))
	# LSTMModel = Sequential()

	# LSTMModel.add(LSTM(4, input_shape=(lookback, 1)))
	# LSTMModel.add(Dense(1))
	# LSTMModel.compile(loss="mean_squared_error", optimizer = 'adam')
	# LSTMModel.fit(XTrainLSTM, newYTrain, epochs = 80, batch_size = 10, verbose = 2)

	# LSTMTrainPredict = LSTMModel.predict(XTrainLSTM)
	# LSTMTestPredict = LSTMModel.predict(XTestLSTM)

	"""
	This is the end of LSTM computation
	"""

	"""
	This is the vanilla RNN implementation
	"""
	lookback = len(newXTrain[0])
	XTrainLSTM = np.reshape(newXTrain, (newXTrain.shape[0], newXTrain.shape[1], 1))
	XTestLSTM = np.reshape(newXTest, (newXTest.shape[0], newXTest.shape[1], 1))
	LSTMModel = Sequential()

	# LSTMModel.add(SimpleRNN(4, input_shape = XTrainLSTM.shape[1:]))
	LSTMModel.add(SimpleRNN(4, return_sequences = True, input_shape = XTrainLSTM.shape[1:]))
	LSTMModel.add(SimpleRNN(4))
	LSTMModel.add(Dense(1))
	LSTMModel.compile(loss="mean_squared_error", optimizer = 'rmsprop')
	LSTMModel.fit(XTrainLSTM, newYTrain, epochs = 70, batch_size = 10, verbose = 2)

	LSTMTrainPredict = LSTMModel.predict(XTrainLSTM)
	LSTMTestPredict = LSTMModel.predict(XTestLSTM)

	"""
	End of vanilla RNN implementation
	"""

	newPredictedYTrain = LSTMTrainPredict.flatten()
	newPredictedYTest = LSTMTestPredict.flatten()


	"""
	This is the original regression part
	"""
	# regression.fit(newXTrain, newYTrain)
	# newPredictedYTrain = regression.predict(newXTrain)
	# newPredictedYTest = regression.predict(newXTest)

	"""
	End of regression part
	"""

	# print "regression coefficients:"
	# print regression.coef_

	newTrainingSize = len(newXTrain)
	newTestSize = len(newXTest)

	comparisonTermListActual = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	comparisonTermListPredict = [[0 for n in range(yearCover)] for i in range(phraseCount)]
	maxTermListActual = [[0 for n in range(yearCover)] for i in range(phraseCount)]


	CVDIFFList = [0 for n in range(len(newYTrain))]

	for sampleIndex in range(newTrainingSize):
		trainingSampleYear = newXTrainYear[sampleIndex]
		trainingSamplePhraseIndex = newXTrainIndices[sampleIndex]
		trainingSamplePhrase = phraseList[int(trainingSamplePhraseIndex)]
		DIFFTermListActual[trainingSamplePhraseIndex][trainingSampleYear] = newYTrain[sampleIndex] - np.max(newXTrain[sampleIndex])
		DIFFTermListPrediction[trainingSamplePhraseIndex][trainingSampleYear] = newPredictedYTrain[sampleIndex] - np.max(newXTrain[sampleIndex])
		maxTermListActual[trainingSamplePhraseIndex][trainingSampleYear] = np.max(newXTrain[sampleIndex])

	for sampleIndex in range(newTestSize):
		testSampleYear = newXTestYear[sampleIndex]
		testSamplePhraseIndex = newXTestIndices[sampleIndex]
		testSamplePhrase = phraseList[int(testSamplePhraseIndex)]
		DIFFTermListActual[testSamplePhraseIndex][testSampleYear] = newYTest[sampleIndex] - np.max(newXTest[sampleIndex])
		DIFFTermListPrediction[testSamplePhraseIndex][testSampleYear] = newPredictedYTest[sampleIndex] - np.max(newXTest[sampleIndex])
		maxTermListActual[testSamplePhraseIndex][testSampleYear] = np.max(newXTest[sampleIndex])

	DIFFTermListActual = np.asarray(DIFFTermListActual)
	DIFFTermListPrediction = np.asarray(DIFFTermListPrediction)
	maxTermListActual = np.asarray(maxTermListActual)

	phraseList = np.asarray(phraseList)

	trendingPhrasesList = []
	trendingPhrasesListWithTrainTest = []
	newTrendingPhraseList = []
	nDCGList = []
	numberOfTrendingWords = 20

	weightDIFF = 0.8
	weightOrigin = 1 - weightDIFF

	comparisonTermListActual = weightOrigin * maxTermListActual + weightDIFF * DIFFTermListActual
	comparisonTermListPredict = weightOrigin * maxTermListActual + weightDIFF * DIFFTermListPrediction


	# comparisonTermListActual = relativeMatrixShortened * comparisonTermListActual
	# comparisonTermListPredict = relativeMatrixShortened * comparisonTermListPredict


	"""
	word embeddings
	"""
	wordembsamples = []


	predefinedSpaceForWriting = 30
	# the TP, FP, TN should all come from the top 20 words (in the slide confirm this)
	for year in range(yearCover):

		currentYearActualDiff = comparisonTermListActual[:, year]
		currentYearPredictDiff = comparisonTermListPredict[:, year]

		actualDistIndices = np.argpartition(currentYearActualDiff, -numberOfTrendingWords)[-numberOfTrendingWords:]
		predictDistIndices = np.argpartition(currentYearPredictDiff, -numberOfTrendingWords)[-numberOfTrendingWords:]

		actualDistIndices = actualDistIndices[np.argsort(currentYearActualDiff[actualDistIndices])]
		predictDistIndices = predictDistIndices[np.argsort(currentYearPredictDiff[predictDistIndices])]

		actualDistIndices = actualDistIndices[::-1]
		predictDistIndices = predictDistIndices[::-1]


		actualTrendingPhrases = phraseList[actualDistIndices]
		predictTrendingPhrases = phraseList[predictDistIndices]

		wordembsamples.append(list(actualTrendingPhrases))

		totalList = []

		# predefine the longest word will be shorter than 20 characters

		for index in range(numberOfTrendingWords):
			currentActualPhrase = actualTrendingPhrases[index]
			currentPredictPhrase = predictTrendingPhrases[index]

			if currentActualPhrase in predictTrendingPhrases:
				numOfSpaces = predefinedSpaceForWriting - len(currentActualPhrase)
				for i in range(numOfSpaces):
					currentActualPhrase += " "
				currentActualPhrase += "+ +"
			else:
				numOfSpaces = predefinedSpaceForWriting - len(currentActualPhrase)
				for i in range(numOfSpaces):
					currentActualPhrase += " "
				currentActualPhrase += "+ -"
			# currentActualPhrase += ("    " + str(currActualTermTopic) + "    " + str(currActualTermTopicOri))

			if currentPredictPhrase in actualTrendingPhrases:
				numOfSpaces = predefinedSpaceForWriting - len(currentPredictPhrase)
				for i in range(numOfSpaces):
					currentPredictPhrase += " "
				currentPredictPhrase += "+ +"
			else:
				numOfSpaces = predefinedSpaceForWriting - len(currentPredictPhrase)
				for i in range(numOfSpaces):
					currentPredictPhrase += " "
				currentPredictPhrase += "- +"
			# currentPredictPhrase += ("    " + str(currPredictTermTopic) + "    " + str(currPredictTermTopicOri))

			totalList.append(currentActualPhrase)
			totalList.append(currentPredictPhrase)

		currentNDCG = calcMRRMAPNDCG(actualDistIndices, predictDistIndices)
		# currentNDCGWrite = str(2004 - yearCover - windowSize + year) + ": " + str(currentNDCG)
		currentNDCGWrite = str(2011 - yearCover + year + 1) + ": " + str(currentNDCG)
		nDCGList.append(currentNDCGWrite)

		newTrendingPhraseList.append("----Trending words " + str(2011 - yearCover + year + 1) + "----actual---predict---topicID---")
		newTrendingPhraseList += totalList
		newTrendingPhraseList.append("\n")

	"""
	Original Word embedding inference part:
	average(topemb1, topemb2, topemb3) -> topemb4
	"""


	# wordEmbTrainingX, wordEmbTrainingY, fullX, fullY = formWordEmbeddingTrainingData(wordembsamples, wordModel, wordModelVocab)
	# wordEmbTrainingYPredict = trainEmbWords(wordEmbTrainingX, wordEmbTrainingY, fullX)

	# wordembsamplesArray = np.asarray(wordembsamples)[3:, :]
	# wordembsamplesArrayPredict = []
	# for wordEmbY in wordEmbTrainingYPredict:
	# 	topKeyphrases = [ele[0] for ele in wordModel.similar_by_vector(wordEmbY, topn = 20)]
	# 	wordembsamplesArrayPredict.append(topKeyphrases)

	# print len(wordembsamplesArray), len(wordembsamplesArray[0])
	# print len(wordembsamplesArrayPredict), len(wordembsamplesArrayPredict[0])

	# wordembsamplesArrayFullY = []
	# for wordEmbY in fullY:
	# 	topKeyphrases = [ele[0] for ele in wordModel.similar_by_vector(wordEmbY, topn = 20)]
	# 	wordembsamplesArrayFullY.append(topKeyphrases)

	# writeWordEmbLists(wordembsamplesArray, wordembsamplesArrayPredict, "wordemb/embskiplists.txt")
	# writeWordEmbLists(wordembsamplesArrayFullY, wordembsamplesArrayPredict, "wordemb/skipcomp.txt")

	"""
	End of original word embedding part
	"""

	writeList(nDCGList, "np-doublegraph/nDCG-soft-nolast-ori.txt")
	writeList(newTrendingPhraseList, "np-doublegraph/soft2RNN4neuron-" + str(windowSize) + "-" + str(weightDIFF) + ".txt")

	pass