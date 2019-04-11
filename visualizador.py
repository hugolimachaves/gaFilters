from deap import algorithms, base, creator, tools
from deap import creator
from deap import tools

import os, random, pickle
import multiprocessing
import numpy as np
import localParameters as lp
import matplotlib.pyplot as plt
from scipy import signal
import random
import json

#
TESTE_MODE = False
ORIGEM = 'siam'
ALVO = 'gt'
GAUSSIAN_DEFINED = False
MEDIA_INICIAL_DEFINED = False
TAMANHO_MEDIA_INICIAL = 15
listaDeVideos = ['bag']#, 'racing', 'ball1', 'octopus', 'bolt2']
SIZE_FILTER = 41
AMPLITUDE_RUIDO = 0.3
dim = 0
N = 41
sigma = N/6
gaussianFilter =  np.array(signal.gaussian(N,sigma))
AMPLITUDE_DE_VISUALIZACAO = 0.25
nome_do_video = 'bag'

'''
**********************************************************************************************
**********************************************************************************************
******************************************* PARTE 1 ******************************************
**********************************************************************************************
**********************************************************************************************
'''


	
if TESTE_MODE:
	dim = 25
	print("A dimensao utilizada para teste sera: ", dim)
	input("Verifique as condiçoes no inicio do codigo")
	
else:
	dim = int(dim)

# Variaveis obtidas atraves do arquivo parameter.json
#--------------------------------------------------------------------
PATH_GT               = lp.getInJson('tracker','gtPath')
PATH_GT_ESPACIAL      = lp.getInJson('tracker','gtEspacialPath')
PATH_SIAMESE          = lp.getInJson('tracker','siamesePath')
PATH_SIAMESE_ESPACIAL = lp.getInJson('tracker','siameseEspacialPath')
SHOW				  = lp.getInJson('tracker','show') 
#--------------------------------------------------------------------




def getPathPickle(tipo):
	if tipo == 'gt':
		caminhoPickle = PATH_GT
	if tipo == 'siam':
		caminhoPickle = PATH_SIAMESE
	if tipo == 'gt_espacial':
		caminhoPickle = PATH_GT_ESPACIAL
	if tipo == 'siam_espacial':
		caminhoPickle = PATH_SIAMESE_ESPACIAL
	return caminhoPickle

	
def getSignal2(tipo, video, xx, yy):

	caminhoPickle = getPathPickle(tipo)
	fullPath = os.path.join(caminhoPickle,video +'.pickle')
	file_siamese = open(fullPath, 'rb')
	list_zFeat = np.array(pickle.load(file_siamese))
	file_siamese.close()
	singleZFeat = np.zeros([int(list_zFeat.shape[0]),256])
	for i in range(xx,xx+1):
		for j in range(yy,yy+1):
			singleZFeat[:,:] = list_zFeat[:,xx,yy,:,0]

	return singleZFeat # t e d, mas apenas uma de x e y

sign = np.zeros(SIZE_FILTER)/SIZE_FILTER

def calcularFFT(sign, escala = 'dB'):
	gamma =  abs(10**-12) # evita log 0
	if escala == 'dB':
		dividendo = abs(np.fft.fftshift(np.fft.fft(sign))) +gamma
		divisor = max(abs(np.fft.fft(sign))) + gamma
		fft = 10*np.log10( dividendo / divisor )
		return fft
	else:
		fft = abs(np.fft.fftshift(np.fft.fft(sign))) 
		return fft

def plotar(sinais,ffts):
	
	sinais = list(sinais)
	ffts = list(ffts)
	nSinais = len(sinais)
	for cont in range(len(sinais)):
		#plot do sinal e da fft 
		plt.subplot(nSinais,2,2*cont+1)
		plt.plot(sinais[cont])
		plt.subplot(nSinais,2,2*cont+2)
		plt.plot(ffts[cont])
	plt.show()

def erroQuadraticoMedio(sinal1, sinal2):
	sinal1 = np.array(sinal1)
	sinal2 = np.array(sinal2)
	erro =  sinal1 - sinal2
	erroQuadratico = erro**2
	MSE =  np.mean(erroQuadratico)
	return MSE

def alinharSinais(sinalMaior, sinalMenor):
	'''
	alinha o sinal maior com o sinal menor. Esta função tira a ultima parte do sinal maior.
	*o erro quadratico medio deve ser 0 ao encontrar o sign impulsivo em um processo de otimizacao
	'''
	#casting de seguranca

	#print("first debug in function: ", sinalMaior.shape)
	sinalMenor =  np.array(sinalMenor)
	sinalMaior =  np.array(sinalMaior)
	'''
	diferencaDimensional = sinalMaior.shape[0] - sinalMenor.shape[0]
	sinalMaior = sinalMaior[: sinalMaior.shape[0] - diferencaDimensional]
	sinalMenor = sinalMenor.ravel(-1)
	sinalMaior = sinalMaior.ravel(-1)
	return sinalMaior,sinalMenor
	'''

	dim_N = np.max(sinalMenor.shape)
	dim_M = np.max(sinalMaior.shape)

	#print("dim_N: ",dim_N, " dim_M: ", dim_M)

	sinalMaior = sinalMaior[ dim_M - dim_N  : dim_M]
	#print("debug in function: sinal maior", sinalMaior.shape, " sinal menor: ", sinalMenor.shape )
	return sinalMaior, sinalMenor

sinal = getSignal2('gt', nome_do_video, 2, 2)

	
SHOW = True

if SHOW:
	dim1 = 0#129
	dim2 = 1#10
	sign1 = sinal[:,dim1]

	axt = plt.subplot(121)
	axt.grid(True)
	axt.set_xlabel('[n] - (frame)')
	axt.set_ylabel('v[n] - (A.U.)')
	axt.set_title('Dimension '+str(dim1))
	#axt.set_ylim([min([0,min(sign1)]),max(sign1)])
	axt.set_ylim([-AMPLITUDE_DE_VISUALIZACAO,AMPLITUDE_DE_VISUALIZACAO])
	#plt.bar(np.arange(0,len(sign1)),sign1)
	plt.plot(sign1)
	plt.plot( [np.mean(sign1) for i in sign1], 'r' )
	plt.plot( [sign1[0] for i in sign1], 'g' )
	
	
	plt.legend(['signal','average','first frame'])
	
	
	sign2 = sinal[:,dim2]
	axt2 = plt.subplot(122)
	axt2.grid(True)
	axt2.set_xlabel('[n] - (frame)')
	axt2.set_ylabel('v[n]  - (A.U.)')
	axt2.set_title('Dimension '+str(dim2))
	#axt2.set_ylim([min([0,min(sign2)]),max(sign2)])
	axt2.set_ylim([-AMPLITUDE_DE_VISUALIZACAO,AMPLITUDE_DE_VISUALIZACAO])
	#plt.bar(np.arange(0,len(sign2)),sign2)
	plt.plot(sign2)

	plt.plot( [np.mean(sign2) for i in sign2], 'r' )
	plt.plot( [sign2[0] for i in sign2], 'g' )
	plt.legend(['signal','average','first frame'])
	plt.suptitle('Descriptor signal', fontsize='large')

	'''
	vetFreq = np.fft.fftshift(abs(np.fft.fft(sign1)))
	axt = plt.subplot(122)
	plt.bar(np.arange(0,len(vetFreq)), vetFreq )
	'''

	
	plt.show()


'''
standard
[100  97  73  66  73  58  49  44  37  39  44  43  41  32  38  50  48  45
  34  34  40  33 -24 -28 -27  33  36  40  39  37  51  57  55  50  53  50
  51  44  45  38  34]

gaussian

[ 72  49  79 -56 -58 -47 -98  74  90  19 -70 -69  92  43 -62 -32  31  70
  82 -75 -66  81 -39 -54 -20 -67  33  50  26  47  84  83 -90 -64 -59 -99
  26  90 -64  83  27]
'''