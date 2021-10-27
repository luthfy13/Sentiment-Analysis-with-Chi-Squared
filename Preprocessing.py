import re
import string
import pandas

def getStopWords():
	df = pandas.read_csv('data/stopwords-en.csv')
	sw = df.word.to_list()
	return sw

def getNegasi():
	df = pandas.read_csv('data/negasi-en.csv')
	neg = df.word.to_list()
	return neg

def hapusStopword(komentar, stopWords):
	querywords = komentar.split()
	resultwords  = [word for word in querywords if word.lower() not in stopWords]
	result = ' '.join(resultwords)
	return result

def tagNegasi(komentar, negasi):
	hsl = " " + komentar + " "
	for word in negasi:
		hsl = re.sub(" " + word + " ", " NOT_", hsl)
	return hsl

def preprocessing(komentar):
	stopWords = getStopWords()
	negasi = getNegasi()

	hsl = re.sub('[\s]+', ' ', str(komentar))
	hsl = hsl.lower() #case folding
	# print(f'========case folding: {hsl}')
	# hsl = hsl.translate(None, string.punctuation) #punctuation removal
	# hsl = hsl.translate(None, string.digits)
	hsl = hsl.translate(str.maketrans('','',string.punctuation))
	hsl = hsl.translate(str.maketrans('','',string.digits))
	# print(f'========puctuation elimination: {hsl}')
	# hsl = gantiKataDasar(hsl) #gnti kata
	hsl = hapusStopword(hsl, stopWords) #hapus stopwords
	# print(f'========stopwords removal: {hsl}')
	hsl = tagNegasi(hsl, negasi) #negation tag
	# print(f'========negation tag: {hsl}')
	# print("\n\n")
	hsl = hsl[1:-1]
	return hsl
