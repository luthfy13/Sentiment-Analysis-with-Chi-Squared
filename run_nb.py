from nb_training import *
from nb_klasifikasi import *
from prettytable import PrettyTable
import pandas
import os


if __name__ == '__main__':
	# df = pandas.read_csv('data/Corpus-en.csv')
	# corpus = df.to_dict('records')
	# print(type(corpus))
	# training(corpus)
	list_sentimen = ['Positif', 'Negatif']
	modelKlasifikasi = getModelKlasifikasi()
	v = getJmlVocabulary()
	jmlFitur = get_jml_fitur_per_sentimen()

	jmlRecordPerSentimen = {}
	for sentimen in list_sentimen:
		jmlRecordPerSentimen[sentimen] = getJmlDokumen(sentimen)

	jmlRecordTraining = getJmlSeluruhDokumen()

	log_prior_prob = getLogPriorProb2(jmlRecordTraining, list_sentimen, jmlRecordPerSentimen)
	
	os.system("cls")
	komentar = str(input("Masukkan komentar: "))
	print(tentukanSentimen(komentar, modelKlasifikasi, v, jmlFitur, log_prior_prob))

	komentar = str(input("Masukkan komentar: "))
	print(tentukanSentimen(komentar, modelKlasifikasi, v, jmlFitur, log_prior_prob))

	komentar = str(input("Masukkan komentar: "))
	print(tentukanSentimen(komentar, modelKlasifikasi, v, jmlFitur, log_prior_prob))