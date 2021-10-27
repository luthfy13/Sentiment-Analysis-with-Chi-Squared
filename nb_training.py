from Preprocessing import *
import string
import math
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import CountVectorizer
import MySQLdb

def clearModel():
	db = MySQLdb.connect("localhost","root", "root","android_sa")
	cs = db.cursor()
	query = "delete from bobot"
	try:
		cs.execute('drop table hsl_prep')
		db.commit()
		cs.execute('drop table fitur')
		db.commit()
		cs.execute('drop table bobot')
		db.commit()

		cs.execute('create table hsl_prep(id int(9) unsigned primary key auto_increment, komentar text, sentimen varchar(50))')
		db.commit()
		cs.execute('create table fitur(id_komentar int(9), id_fitur int(9), fitur varchar(500), primary key(id_komentar, id_fitur))')
		db.commit()
		cs.execute("create table bobot(id int(9) unsigned primary key auto_increment, fitur varchar(500), kelas enum('Positif','Negatif'), cond_prob decimal(20,16))")
		db.commit()
	except:
		print("hapus data gagal")
		db.rollback()
	db.close

def getPrepCorpus(corpus):
	rawDatas = corpus
	for i in range(0 , len(rawDatas)):
		rawDatas[i]['Komentar'] = preprocessing(rawDatas[i]['Komentar'])
	return rawDatas

def insertPrepCorpus(prepCorpus):
	query = []
	query.append("insert into hsl_prep values")
	i=1
	for komentar in prepCorpus:
		query.append("('%i', '%s', '%s')," % (i, komentar['Komentar'], komentar['Sentimen']))
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

def getNGram(text, ngram):
	texts = []
	texts.append(text)
	vect = CountVectorizer(ngram_range=(ngram,ngram))
	X_dtm = vect.fit_transform(texts)
	X_dtm = X_dtm.toarray()
	return vect.get_feature_names()

def getListFitur(prepCorpus):
	listFitur = []
	idKomentar=1
	idFitur=1
	for record in prepCorpus:
		# komentar = record['Komentar'].split()
		try:
			komentar = getNGram(record['Komentar'], 2)
		except:
			komentar = record['Komentar'].split()
		for kata in komentar:
			fitur = {}
			fitur['idKomentar'] = idKomentar
			fitur['id'] = idFitur
			fitur['fitur'] = kata
			listFitur.append(fitur)
			idFitur = idFitur + 1
		idKomentar = idKomentar + 1
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

def getFiturUnik():
	db = MySQLdb.connect("localhost","root", "root","android_sa")
	crFeatureList = db.cursor()
	qFeatureList = "select distinct fitur from fitur order by fitur asc"
	try:
		crFeatureList.execute(qFeatureList)
		rsFeatureList = crFeatureList.fetchall()
		featureList = []
		for row in rsFeatureList:
			featureList.append(row[0])
	except:
		print("baca data fitur gagal")
	db.close
	return featureList

def getFreqFitur():
	db = MySQLdb.connect("localhost","root", "root","android_sa")
	cs = db.cursor()
	query = "SELECT fitur.fitur, COUNT(fitur.fitur) AS frekuensi, hsl_prep.sentimen \
			FROM fitur INNER JOIN hsl_prep ON fitur.id_komentar = hsl_prep.id \
			GROUP BY hsl_prep.sentimen, fitur.fitur ORDER BY fitur.fitur ASC"
	try:
		cs.execute(query)
		rs = cs.fetchall()
		jml = 0
		fitur = ''
		sentimen = ''
		dataFrekFitur = {}
		for row in rs:
			fitur    = row[0]
			jml      = row[1]
			sentimen = row[2]
			dataFrekFitur[fitur,sentimen] = jml
	except:
		print("baca data doc gagal")
	db.close
	for sentimen in ['Positif', 'Negatif']:
		for fitur in getFiturUnik():
			try:
				dataFrekFitur[fitur,sentimen] = dataFrekFitur[fitur,sentimen]
			except:
				dataFrekFitur[fitur,sentimen] = 0
	return dataFrekFitur

def getJmlVocabulary():
	db = MySQLdb.connect("localhost","root", "root","android_sa")
	cs = db.cursor()
	query = "SELECT count(DISTINCT fitur) FROM fitur"
	try:
		cs.execute(query)
		rs = cs.fetchall()
		jml = 0
		for row in rs:
			jml = row[0]
	except:
		print("baca data doc gagal")
	db.close
	return jml

def getJmlFitur(sentimen):
	db = MySQLdb.connect("localhost","root", "root","android_sa")
	cs = db.cursor()
	query = "SELECT count(fitur.fitur), hsl_prep.sentimen FROM fitur \
			INNER JOIN hsl_prep ON fitur.id_komentar = hsl_prep.id \
			WHERE hsl_prep.sentimen = '%s'" % sentimen
	try:
		cs.execute(query)
		rs = cs.fetchall()
		jml = 0
		for row in rs:
			jml = row[0]
	except:
		print("baca data gagal")
	db.close
	return jml

def pembobotan():
	alpha = 1
	logCondProb={}
	dataFrekFitur = getFreqFitur()
	jmlFitur = {}
	v = getJmlVocabulary()
	for sentimen in ['Positif', 'Negatif']:
		jmlFitur[sentimen] = getJmlFitur(sentimen)
		for fitur in getFiturUnik():
			logCondProb[fitur,sentimen] = math.log(float(dataFrekFitur[fitur,sentimen] + alpha) / float(jmlFitur[sentimen] + (v*alpha)), 2)
			# pakai m-estimate, m = 1
			# logCondProb[fitur,sentimen] = math.log(float(dataFrekFitur[fitur,sentimen] + (1*0.3)) / float(jmlFitur[sentimen] + 1), 2)
	return logCondProb

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
	# df_low_score = pandas.DataFrame(low_score_features, columns=["colummn"])
	# df_low_score.to_csv('ListLowScore.csv', index=False)

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
	clearModel()
	prepCorpus = getPrepCorpus(corpus)
	prepCorpus = chi_square(prepCorpus)
	insertPrepCorpus(prepCorpus)
	insertListFitur(prepCorpus)
	logCondProb = pembobotan()
	query = []
	query.append("insert into bobot (fitur,kelas,cond_prob) values")
	for sentimen in ['Positif', 'Negatif']:
		for fitur in getFiturUnik():
			query.append("('%s', '%s', %.16f)," % (fitur, sentimen, logCondProb[fitur,sentimen]) )
	finalQuery = ''.join(query)
	finalQuery = finalQuery[:-1]
	
	db = MySQLdb.connect("localhost","root", "root","android_sa")
	cs = db.cursor()
	try:
		cs.execute(finalQuery)
		db.commit()
	except:
		print("insert data gagal")
		db.rollback()
	db.close
