from Preprocessing import *
import math
from sklearn.feature_extraction.text import CountVectorizer
import MySQLdb

def getJmlDokumen(sentimen):
	db = MySQLdb.connect("localhost","root", "root","android_sa")
	cs = db.cursor()
	query = "select count(komentar) from hsl_prep where sentimen = '%s' " % sentimen
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

def getJmlSeluruhDokumen():
	db = MySQLdb.connect("localhost","root", "root","android_sa")
	cs = db.cursor()
	query = "select count(komentar) from hsl_prep"
	try:
		cs.execute(query)
		rs = cs.fetchall()
		jml = 0
		for row in rs:
			jml = row[0]
	except:
		print("baca data all doc gagal")
	db.close
	return jml

def getLogPriorProb(sentimen):
	N_kelas = float(getJmlDokumen(sentimen))
	N = float(getJmlSeluruhDokumen())
	return math.log(N_kelas/N, 2)

def getLogPriorProb2(jmlRecordTraining, list_sentimen, jmlRecordPerSentimen):
	log_prior_prob = {}
	for sentimen in list_sentimen:
		log_prior_prob[sentimen] = math.log(jmlRecordPerSentimen[sentimen]/jmlRecordTraining, 2)
	return log_prior_prob


def getModelKlasifikasi():
	model = {}
	db = MySQLdb.connect("localhost","root", "root","android_sa")
	cs = db.cursor()
	query = "SELECT fitur, kelas, cond_prob FROM bobot"
	try:
		cs.execute(query)
		rs = cs.fetchall()
		fitur = ''
		sentimen = ''
		bobot = 0.0
		for row in rs:
			fitur = row[0]
			sentimen = row[1]
			bobot = row[2]
			model[fitur,sentimen] = bobot
	except:
		print("baca data doc gagal")
	db.close
	return model

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

def get_jml_fitur_per_sentimen():
	jml_fitur = {}
	for sentimen in ['Positif', 'Negatif']:
		jml_fitur[sentimen] = getJmlFitur(sentimen)
	return jml_fitur

def tentukanSentimen(komentar, modelKlasifikasi):
	alpha = 1
	print("alpha")
	probAkhir = {}
	komentar = preprocessing(komentar)
	arrayKata = komentar.split()
	print("split komentar")
	jmlFitur = {}
	v = getJmlVocabulary()
	print("getJmlVocabulary")
	for sentimen in ['Positif', 'Negatif']:
		jmlFitur[sentimen] = getJmlFitur(sentimen)
		probAkhir[sentimen] = float(getLogPriorProb(sentimen))
		for fitur in arrayKata:
			try:
				probAkhir[sentimen] = float(probAkhir[sentimen]) + float(modelKlasifikasi[fitur,sentimen])
			except:
				probAkhir[sentimen] = float(probAkhir[sentimen]) + (math.log(float(alpha) / float(jmlFitur[sentimen] + (v*alpha)), 2))
				# pakai m-estimate, m = 3
				# probAkhir[sentimen] = float(probAkhir[sentimen]) + (math.log(float(3*0.3) / float(jmlFitur[sentimen] + 3), 2))
	temp = -9999
	hsl = ''
	for sentimen in ['Positif', 'Negatif']:
		if (probAkhir[sentimen] > temp):
			hsl = sentimen
		temp = probAkhir[hsl]
	return hsl

def getNGram(text, ngram):
	texts = []
	texts.append(text)
	vect = CountVectorizer(ngram_range=(ngram,ngram))
	X_dtm = vect.fit_transform(texts)
	X_dtm = X_dtm.toarray()
	return vect.get_feature_names()

def getLowScoreFeatures():
	df = pandas.read_csv('ListLowScore.csv')
	lsf = df.colummn.to_list()
	return lsf

def tentukanSentimen(komentar, modelKlasifikasi, v, jmlFitur, logPriorProb):
	alpha = 1
	probAkhir = {}
	komentar = preprocessing(komentar)
	# arrayKata = komentar.split()
	try:
		arrayKata = getNGram(komentar, 2)
	except:
		arrayKata = komentar.split()
	
	for sentimen in ['Positif', 'Negatif']:
		probAkhir[sentimen] = float(logPriorProb[sentimen])
		for fitur in arrayKata:
			try:
				probAkhir[sentimen] = float(probAkhir[sentimen]) + float(modelKlasifikasi[fitur,sentimen])
			except:
				probAkhir[sentimen] = float(probAkhir[sentimen]) + (math.log(float(alpha) / float(jmlFitur[sentimen] + (v*alpha)), 2))
				# pakai m-estimate, m = 3
				# probAkhir[sentimen] = float(probAkhir[sentimen]) + (math.log(float(3*0.3) / float(jmlFitur[sentimen] + 3), 2))
	temp = -9999
	hsl = ''
	for sentimen in ['Positif', 'Negatif']:
		if (probAkhir[sentimen] > temp):
			hsl = sentimen
		temp = probAkhir[hsl]
	return hsl

# if __name__ == '__main__':
# 	record = {}
# 	record['Komentar'] = 'aplikasinya sangat jelek'
# 	record['Sentimen'] = 'Crash'
# 	print tentukanSentimen(record['Komentar']) == record['Sentimen']