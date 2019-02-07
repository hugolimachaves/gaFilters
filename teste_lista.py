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

'''
**********************************************************************************************
**********************************************************************************************
******************************************* PARTE 1 ******************************************
**********************************************************************************************
**********************************************************************************************
'''

def overall(DIM):
	
	DIM = int(DIM)
	# Variaveis obtidas atraves do arquivo parameter.json
	#--------------------------------------------------------------------
	PATH_GT               = lp.getInJson('tracker','gtPath')
	PATH_GT_ESPACIAL      = lp.getInJson('tracker','gtEspacialPath')
	PATH_SIAMESE          = lp.getInJson('tracker','siamesePath')
	PATH_SIAMESE_ESPACIAL = lp.getInJson('tracker','siameseEspacialPath')
	SHOW				  = lp.getInJson('tracker','show') 
	#--------------------------------------------------------------------

	#TIPO = 'siam'
	#FEATURE =  30
	#VIDEO = 'bag'
	X =3
	Y =3
	#DIM = 25
	SIZE_FILTER = 41

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

	def getSignal(tipo, video, xx, yy):
		
		caminhoPickle = getPathPickle(tipo)
		fullPath = os.path.join(caminhoPickle,tipo+'_'+video +'.pickle')
		file_siamese = open(fullPath, 'rb')
		list_zFeat = np.array(pickle.load(file_siamese))
		file_siamese.close()
		media = []
		pontos_de_recorte = [i for i in range(0,int(list_zFeat.shape[0]+1),int(list_zFeat.shape[0]/9))]
		MEDIA_ESPACIAL = 1
		for i in range(MEDIA_ESPACIAL):
			media.append(list_zFeat[pontos_de_recorte[i]:pontos_de_recorte[i+1],:,:,:,:])
		media = np.array(media)
		result = np.mean(media,axis=0)
		acc = np.zeros([int(list_zFeat.shape[0]/9),256])
		for i in range(xx,xx+1):
			for j in range(yy,yy+1):
				acc[:,:] = result[:,xx,yy,:,0] + acc[:,:]
		return acc
		
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

	filtro = np.zeros(SIZE_FILTER)/SIZE_FILTER

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
		*o erro quadratico medio deve ser 0 ao encontrar o filtro impulsivo em um processo de otimizacao
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

	'''
	**********************************************************************************************
	**********************************************************************************************
	******************************************* PARTE 2 ******************************************
	**********************************************************************************************
	**********************************************************************************************
	'''

	import random

	from deap import base
	from deap import creator
	from deap import tools

	GAMMA = 10**-10
	MENOR_AMPLITUDE = -100
	MAIOR_AMPLITUDE = 100
	PROBABILIDADE_MUTACAO_GENE = 1/SIZE_FILTER
	PROBABILIDADE_MUTACAO_INDIVIDUO = 0.9
	PROBABILIDADE_CROSS_OVER=0.95
	POPULACAO = 5*SIZE_FILTER
	PORCENTAGEM = 10
	TAMANHO_DO_TORNEIO = round(POPULACAO*PORCENTAGEM/100)
	MAX_ITE = 300

	creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
	creator.create("Individual", list, fitness=creator.FitnessMin)

	toolbox = base.Toolbox()
	# Attribute generator 
	toolbox.register("attr_int", random.randint, MENOR_AMPLITUDE, MAIOR_AMPLITUDE) # nosso filtro vai ter uma resolução de apenas 1000 valores
	# Structure initializers
	toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, SIZE_FILTER) # teremos um filtro de 41 dimensoes
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)

	sinal_siamese_master = []
	sinal_gt_master = []
	listaDeVideos = ['bag'] #, 'tunnel', 'ball1']

	for video in range(len(listaDeVideos)):
		sinal_siamese_master.append([])
		sinal_gt_master.append([])
		for xx in range(6):
			sinal_siamese_master[video].append([])
			sinal_gt_master[video].append([])
			for yy in range(6):
				sinal_siamese_master[video][xx].append([])
				sinal_gt_master[video][xx].append([])


	for numero_video,video in enumerate(listaDeVideos):
		for xx in range(6):
			for yy in range(6):

				fullSignalSiam = getSignal2('siam',video,xx,yy)
				sinal_siamese = np.ravel(fullSignalSiam[:,DIM])
				sinal_siamese_master[numero_video][xx][yy] = np.array(sinal_siamese)
				sinal_siamese_master[numero_video][xx][yy] = np.array(sinal_siamese) + (np.random.rand(len(sinal_siamese_master[numero_video][xx][yy]))-0.5)*0.3

				fullSignalGt = getSignal2('gt',video,xx,yy)
				sinal_gt = np.ravel(fullSignalGt[:,DIM])
				sinal_gt_master[numero_video][xx][yy] = np.array(sinal_gt)

	'''
	sen_exp = np.sin([ (i/4) for i in range(196)]) * np.array([ (np.exp(-0.02*i)) for i in range(1,197)])
	print(sen_exp)
	sinal_siamese_master = np.array( sen_exp + np.sin([ i/7 for i in range(196)]))
	sinal_gt_master = np.sin([ (i/4) for i in range(196)])
	'''

	#print('done')
	#print(sinal_gt_master[0][0])

	def evalFilter(individual):
		individual = [float(i) for i in individual]
		filtro = np.array(individual)/sum([abs(i)+GAMMA for i in individual])
		corr = signal.correlate(sinal_siamese_master, filtro, mode='same')
		#sinal,corr = alinharSinais(sinal_siamese_master,corr)
		#MSE = erroQuadraticoMedio(sinal,corr)
		#return MSE,


	def evalFilter2(individual):
		individual = [float(i) for i in individual]
		
		filtro = np.array(individual)/sum([abs(i)+GAMMA for i in individual])
		MSE = 0

		for video in range(len(listaDeVideos)):
			for xx in range(6):
				for yy in range(6):
					
					#corr = signal.convolve(sinal_siamese_master[xx][yy],filtro,  mode='full') # alterar para correlate!		
					#sinal,corr = alinharSinais(corr,sinal_gt_master[xx][yy])
					corr = np.correlate(sinal_siamese_master[video][xx][yy],filtro,  mode='full') # alterar para correlate!
					sinal,corr = alinharSinais(corr,sinal_gt_master[video][xx][yy])


					#print("sinal e corr: ",sinal.shape, corr.shape)
					MSE += erroQuadraticoMedio(corr,sinal)
		return MSE,

	'''
	#setado para o caso real
	'''

	toolbox.register("evaluate", evalFilter2)
	toolbox.register("mate", tools.cxTwoPoint)
	toolbox.register("mutate", tools.mutUniformInt ,indpb=PROBABILIDADE_MUTACAO_GENE,low= MENOR_AMPLITUDE, up=MAIOR_AMPLITUDE )#, MENOR_AMPLITUDE, MAIOR_AMPLITUDE,indpb=0.05 )
	toolbox.register("select", tools.selTournament, tournsize=TAMANHO_DO_TORNEIO)

	hof = tools.HallOfFame(10)

	def main():
		pop = toolbox.population(n=POPULACAO) # nossa população inicial sera de 300
		
		# Evaluate the entire population
		fitnesses = list(map(toolbox.evaluate, pop))
		
		for ind, fit in zip(pop, fitnesses):
			
			ind.fitness.values = fit

		CXPB = PROBABILIDADE_CROSS_OVER
		MUTPB = PROBABILIDADE_MUTACAO_INDIVIDUO

		# Extracting all the fitnesses of 
		fits = [ind.fitness.values[0] for ind in pop]

		# Variable keeping track of the number of generations
		g = 0

		# Begin the evolution
		while  g < MAX_ITE:
			# A new generation
			g = g + 1
			print("-- Generation %i --" % g)

			# Select the next generation individuals
			offspring = toolbox.select(pop ,len(pop))
			'''
			#print("Numero de individuos: ",len(offspring))
			#print(offspring)
			#input('aperta qualquer coisa')
			'''
			# Clone the selected individuals
			offspring = list(map(toolbox.clone, offspring))

			# Apply crossover and mutation on the offspring
			for child1, child2 in zip(offspring[::2], offspring[1::2]):
				if random.random() < CXPB:
					
					#print('individuos cruzados: ',toolbox.mate(child1, child2))
					del child1.fitness.values
					del child2.fitness.values

			for mutant in offspring:
				if random.random() < MUTPB:
					toolbox.mutate(mutant)
					del mutant.fitness.values

			# Evaluate the individuals with an invalid fitness
			invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

			fitnesses = map(toolbox.evaluate, invalid_ind)
			for ind, fit in zip(invalid_ind, fitnesses):
				ind.fitness.values = fit
			#print("populacao de pais: ", pop)
			hof.update(pop)
			pop[:] = offspring

			# Gather all the fitnesses in one list and print the stats
			fits = [ind.fitness.values[0] for ind in pop]
			
			#print(hof[3])

			length = len(pop)
			mean = sum(fits) / length
			sum2 = sum(x*x for x in fits)
			std = abs(sum2 / length - mean**2)**0.5
			
			#print("  Min %s" % min(fits))
			#print("  Max %s" % max(fits))
			#print("  Avg %s" % mean)
			#print("  Std %s" % std)
		return hof[0]
	vet = main()
	
	arquivo = os.path.join(lp.getInJson("tracker","filterFolder"), "filtro_"+str(DIM)+".pickle" )
	pickle_var = open(arquivo ,"wb")
	pickle.dump(np.array(vet), pickle_var)
	pickle_var.close()
	return np.array(vet)







'''
if SHOW:
	filtro = vet
	axt = plt.subplot(121)
	axt.set_ylim([min([0,min(filtro)]),max(filtro)])
	plt.bar(np.arange(0,len(filtro)),filtro)
	vetFreq = np.fft.fftshift(abs(np.fft.fft(vet)))
	axt = plt.subplot(122)
	plt.bar(np.arange(0,len(vetFreq)), vetFreq )
	plt.show()
'''	