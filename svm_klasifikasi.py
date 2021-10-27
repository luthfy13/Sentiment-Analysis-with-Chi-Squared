from libsvm.svmutil import *
from Preprocessing import *
from sklearn.feature_extraction.text import CountVectorizer

def getListFiturUnik():
	df = pandas.read_csv('ListFiturUnik.csv')
	lfu = df.colummn.to_list()
	return lfu

def splitData(komentar):
	hasil = []
	for komen in komentar:
		# kata = komen.split()
		kata = getNGram(komen,2)
		hasil.append(kata)
	return hasil

def getSVMFeatureVector(komentar, listFiturUnik):
	sortedFeatures = sorted(listFiturUnik)
	map = {}
	featureVector = []
	for t in komentar:
		label = 0
		map = {}
		for w in sortedFeatures:
			map[w] = 0
		for word in t:
			if word in map:
				map[word] = 1
		values = list(map.values())
		featureVector.append(values)                    
	return featureVector

def getNGram(text, ngram):
	texts = []
	texts.append(text)
	vect = CountVectorizer(ngram_range=(ngram,ngram))
	X_dtm = vect.fit_transform(texts)
	X_dtm = X_dtm.toarray()
	return vect.get_feature_names()

def tentukanSentimenSVM(model, komentar, listFiturUnik):
	komentar2 = []
	hasil = ''
	komentar = preprocessing(komentar)
	komentar2.append(komentar)
	splittedData = []
	splittedData = splitData(komentar2)
	komentarSVM = getSVMFeatureVector(splittedData, listFiturUnik)
	p_labels, p_accs, p_vals = svm_predict([0] * len(komentarSVM), komentarSVM, model, "-q")
	if (p_labels[0]==0.0):
		hasil = "Positif"
	elif (p_labels[0]==1.0):
		hasil = "Negatif"
	return hasil
