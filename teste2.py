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


def alinharSinais(sinalMaior, sinalMenor):
	'''
	alinha o sinal maior com o sinal menor. Esta função tira a ultima parte do sinal maior.
	*o erro quadratico medio deve ser 0 ao encontrar o filtro impulsivo em um processo de otimizacao
	'''
	#casting de seguranca
	sinalMenor =  np.array(sinalMenor)
	sinalMaior =  np.array(sinalMaior)
	diferencaDimensional = sinalMaior.shape[0] - sinalMenor.shape[0]
	sinalMaior = sinalMaior[: sinalMaior.shape[0] - diferencaDimensional]
	sinalMenor = sinalMenor.ravel(-1)
	sinalMaior = sinalMaior.ravel(-1)
	return sinalMaior,sinalMenor

#sinal = np.array([0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,1,1,1,1,0.5,0.5,0.5,0.5,1,1,1,1,0.5,0.5,0.5,0.5,1,1,1,1,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5])
sinal = np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
#impulso= np.array([1,7,9,10,10,10,7,5,3,1,1,0.5,.3,0])
#impulso = np.flip(np.array([0,0,0,0,0,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0,0,0]),0)
impulso = np.flip(np.array([0.05,0,0,0.12,0,0,0.25,0,0,0.5,0,0,1]),0)
corr = signal.convolve( impulso,sinal,   mode='full')
#sinal,corr = alinharSinais(sinal,corr)

plt.subplot(1,3,1)
plt.bar(np.arange(0,len(sinal)), sinal)
plt.subplot(1,3,2)
plt.bar(np.arange(0,len(impulso)),impulso)
plt.subplot(1,3,3)
plt.bar(np.arange(0,len(corr)),corr)
plt.show()