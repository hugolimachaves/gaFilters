from multiprocessing import Pool
import teste_lista as tl
import time 
import localParameters as lp
import json
import os






def f(x):
	
	x = str(x)
	k = tl.overall(x)
	print(k)
	return k	


if __name__ == '__main__':
	fileName = 'version.json'
	KEY = 'version'

	with open(fileName) as file:
		data = json.load(file)
	data[KEY] += 1

	os.mkdir(os.path.join(lp.getInJson("tracker","filterFolder"),'filtro_v'+str(data['version'])))

	with open(fileName,'w') as file:
		json.dump(data,file)

	N_THREADS = lp.getInJson("tracker","gaNumProcess")
	N_FILTROS = 256

	#input("Verifique as condiçoes no inicio do codigo")
	p = Pool(N_THREADS)
	listaDim = [i for i in range(0,N_FILTROS,N_THREADS)]
	print(listaDim)
	
	for i in listaDim:
		listaExec = []
		for j in range(N_THREADS):
			listaExec.append(i+j)
			
		print('Nesta rodada serao executadas otimizacoes de filtros  das seguintes dimensoes: ', listaExec)
		p.map(f, listaExec)
		
