from svm_training import *
from svm_klasifikasi import *
from prettytable import PrettyTable
import pandas
import os

if __name__ == '__main__':

	df = pandas.read_csv('data/Corpus-en-test.csv')
	corpus = df.to_dict('records')
	corpus = sorted(corpus, key = lambda i: i['Sentimen'], reverse=True)
	training(corpus)
	
	# model = svm_load_model('svm_model')
	# os.system("cls")
	# komentar = raw_input("Masukkan komentar: ")
	# hasil = tentukanSentimenSVM(model, komentar)
	# print hasil