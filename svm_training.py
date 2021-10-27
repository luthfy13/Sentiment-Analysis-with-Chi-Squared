from libsvm.svmutil import *
from Preprocessing import *
import pandas
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
import json

def getPrepCorpus(corpus):
	rawDatas = corpus
	for i in range(0 , len(rawDatas)):
		rawDatas[i]['Komentar'] = preprocessing(rawDatas[i]['Komentar'])
	return rawDatas


def getListFitur(prepCorpus):
	listFitur = []
	for record in prepCorpus:
		# komentar = record['Komentar'].split()
		komentar = getNGram(record['Komentar'], 2)
		for kata in komentar:
			listFitur.append(kata)
	return listFitur

def insertListFitur(prepCorpus):
	query = []
	query.append("insert into fitur values")
	i=1
	for kata in getListFitur(prepCorpus):
		query.append("('%i', '%i', '%s')," % (kata['idKomentar'], kata['id'], kata['fitur']))
		i = i+1
	finalQuery = ''.join(query)[:-1]
	
	db = MySQLdb.connect("localhost","root", "root","android_sa")
	cs = db.cursor()
	try:
		cs.execute(finalQuery)
		db.commit()
	except TypeError as e:
		print("insert data gagal: " + e)
		db.rollback()
	db.close

def getFiturUnik(listFitur):
	return list(set(listFitur))

def getSVMFeatureVectorAndLabels(dataSet, fiturUnik):
	sortedFeatures = sorted(fiturUnik)
	map = {}
	feature_vector = []
	labels = []

	for t in dataSet:
		label = 0
		map = {}
		for w in sortedFeatures:
			map[w] = 0
		kata = t[0]
		tweet_opinion = t[1]
		for word in kata:
			if word in map:
				map[word] = 1
		values = list(map.values())
		feature_vector.append(values)
		if(tweet_opinion == 'Positif'):
			label = 0
		elif(tweet_opinion == 'Negatif'):
			label = 1
		labels.append(label)

	return feature_vector, labels

def makeDataSet(trainingCorpus):
	dataSet = []
	for i in range(0, len(trainingCorpus)):
		# dataSet.append((trainingCorpus[i]['Komentar'].split(), trainingCorpus[i]['Sentimen']))
		dataSet.append((getNGram(trainingCorpus[i]['Komentar'], 2), trainingCorpus[i]['Sentimen']))
	return dataSet

def getNGram(text, ngram):
	texts = []
	texts.append(text)
	vect = CountVectorizer(ngram_range=(ngram,ngram))
	X_dtm = vect.fit_transform(texts)
	X_dtm = X_dtm.toarray()
	return vect.get_feature_names()

def chi_square(corpus):
	df_corpus = pandas.DataFrame(corpus)

	X = df_corpus.Komentar.tolist()
	y = df_corpus.Sentimen.tolist()

	vect  = CountVectorizer(ngram_range=(2,2))
	X_dtm = vect.fit_transform(X)
	X_dtm = X_dtm.toarray()

	chisq_score = chi2(X_dtm, y)[0]

	df_hasil = pandas.DataFrame({'fitur': vect.get_feature_names(), 'chi2_score': chisq_score})
	df_hasil = df_hasil[df_hasil['chi2_score'] == 0]

	low_score_features = df_hasil.fitur.tolist()
	df_low_score = pandas.DataFrame(low_score_features, columns=["colummn"])
	df_low_score.to_csv('ListLowScore.csv', index=False)

	for i in range (0, len(X)):
		array_kata = getNGram(X[i], 2)
		# array_kata = X[i].split()
		array_kata_filtered = [word for word in array_kata if word.lower() not in low_score_features]
		X[i] = ' '.join(array_kata_filtered)

	final_corpus = []
	for i in range(0, len(X)):
		k = {}
		k['Komentar'] = X[i]
		k['Sentimen'] = y[i]
		final_corpus.append(k)

	return final_corpus

def training(corpus):
	LINEAR_KERNEL = 0
	prepCorpus = getPrepCorpus(corpus)
	prepCorpus = chi_square(prepCorpus)
	listFitur = getListFitur(prepCorpus)
	fiturUnik = getFiturUnik(listFitur)
	df_fitur_unik = pandas.DataFrame(fiturUnik, columns=["colummn"])
	df_fitur_unik.to_csv('ListFiturUnik.csv', index=False)
	dataSet = makeDataSet(prepCorpus)
	feature_vectors, labels = getSVMFeatureVectorAndLabels(dataSet, fiturUnik)
	problem = svm_problem(labels, feature_vectors)
	param = svm_parameter('-q')
	param.kernel_type = LINEAR
	svmClassifier = svm_train(problem, param)


	svm_save_model('svm_model', svmClassifier)

	return svmClassifier
