from nb_training import *
from nb_klasifikasi import *
from prettytable import PrettyTable
import pandas
import os


def evaluasi(partisiTraining, partisiTesting):
	training(partisiTraining)
	print("Training Data Finish")
	
	list_sentimen = ['Positif', 'Negatif']
	modelKlasifikasi = getModelKlasifikasi()
	v = getJmlVocabulary()
	jmlFitur = get_jml_fitur_per_sentimen()
	# lowScoreFeatures = getLowScoreFeatures()

	jmlRecordPerSentimen = {}
	for sentimen in list_sentimen:
		jmlRecordPerSentimen[sentimen] = getJmlDokumen(sentimen)
	jmlRecordTraining = getJmlSeluruhDokumen()
	log_prior_prob = getLogPriorProb2(jmlRecordTraining, list_sentimen, jmlRecordPerSentimen)

	y = 0

	pos_pos = 0
	pos_neg = 0
	neg_pos = 0
	neg_neg = 0

	for x in range(0, len(partisiTesting)):
		# print(f'{partisiTesting[x]["Komentar"]}', end=" ")
		res_sentimen = tentukanSentimen(partisiTesting[x]['Komentar'], modelKlasifikasi, v, jmlFitur, log_prior_prob)
		if res_sentimen == partisiTesting[x]['Sentimen']:
			y = y+1
			if partisiTesting[x]['Sentimen'] == 'Positif':
				pos_pos = pos_pos+1
			elif partisiTesting[x]['Sentimen'] == 'Negatif':
				neg_neg = neg_neg+1
		else:
			if partisiTesting[x]['Sentimen'] == 'Positif':
				if res_sentimen == 'Negatif':
					pos_neg = pos_neg+1
			elif partisiTesting[x]['Sentimen'] == 'Negatif':
				if res_sentimen == 'Positif':
					neg_pos = neg_pos+1
		str_tampil = "komentar diproses:" + str(x+1).rjust(8)
		print(str_tampil, end='')
		print('\b' * len(str_tampil), end='', flush=True)

	t = PrettyTable(['', 'Response Positif', 'Response Negatif'])
	t.align['']               = 'l'
	t.align['Response Positif'] = 'c'
	t.align['Response Negatif'] = 'c'
	t.align['Response Crash']   = 'c'
	
	t.add_row(['Reference Positif', "TP = " + str(pos_pos).rjust(5), "FN = " + str(pos_neg).rjust(5)])
	t.add_row(['Reference Negatif', "FP = " + str(neg_pos).rjust(5), "TN = " + str(neg_neg).rjust(5)])
	print(t)

	print('Jumlah data benar = %i dari %i data' % (y, len(partisiTesting)))
	akurasi = (float(y)/len(partisiTesting) * 100.0)
	print('Akurasi           = %.2f%s' % (akurasi, '%'))
	
	return akurasi

def bacaCorpus():
	df = pandas.read_csv('data/Corpus-en-small.csv')
	corpus = df.to_dict('records')
	corpus = sorted(corpus, key = lambda i: i['Sentimen'], reverse=True)
	return corpus

def bagiPartisi():
	corpus = bacaCorpus()
	sentimentList = ['Positif', 'Negatif']
	jmlKelas = len(sentimentList)
	jmlRecord = len(corpus)                           #20000

	K = 4
	jmlDataPerKelas    = jmlRecord/jmlKelas           #10000
	jmlDataPerPartisi  = jmlRecord/K                  #5000
	jmlDataPerSentimen = jmlDataPerPartisi/jmlKelas   #2500

	list_partisi = []
	for i in range (0, K):
		init = jmlDataPerSentimen*i
		j = int(init)
		k = int(init+jmlDataPerSentimen)
		partisi = []
		for i in range (j, k):
			partisi.append(corpus[i])

		j = int(init+jmlDataPerKelas)
		k = int(j+jmlDataPerSentimen)
		for i in range (j, k):
			partisi.append(corpus[i])

		list_partisi.append(partisi)
	return list_partisi

# algorithm inventor: me
def get_list_index(K):
	list_indexes = []
	for i in range(0, K):
		x=0
		list_index = []
		for j in range(i, K):
			list_index.append(j)
			x += 1
		if x < K:
			for y in range(0, K-x):
				list_index.append(y)
		list_indexes.append(list_index)
	return list_indexes

if __name__ == '__main__':
	K = 4

	hslIterasi=[]

	os.system("cls")
	print("Evaluasi Naive Bayes Classifier: K-fold Cross-validation")
	print("========================================================")
	print("")
	print(f"Jumlah keseluruhan data  = {len(bacaCorpus())}")
	print(f"Nilai K                  = {K}")
	print(f"Jumlah partisi           = {K}")
	print(f"Jumlah data tiap partisi = {int(len(bacaCorpus())/K)}")
	print(f"Fitur                    = Bigram")
	print(f"Feature Selection        = Chi-Square Feature Selection")
	print(f"")

	list_index = get_list_index(K)
	list_partisi = bagiPartisi()

	for i in range(0, K):
		p = list_index[i][0]
		q = list_index[i][1]
		r = list_index[i][2]
		s = list_index[i][3]

		print(f"Iterasi ke-{i+1}:")
		print(f"Partisi Training = Partisi {p+1} + Partisi {q+1} + Partisi {r+1}")
		print(f"Partisi Testing  = Partisi {s+1}")
		
		partisiTraining = []
		partisiTesting = []

		partisiTraining = list_partisi[p] + list_partisi[q] + list_partisi[r]
		partisiTesting  = list_partisi[s]

		hslIterasi.append(evaluasi(partisiTraining, partisiTesting))
		print(f"Iterasi ke-{i+1} selesai...")
		print("")

	akurasi = sum(hslIterasi) / 4.0
	print('Akurasi Classifier = %.2f%s' % (akurasi, '%'))
	input("Evaluasi Classifier selesai...")
