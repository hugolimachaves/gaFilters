import json
import socket


def readParameters():
	with open('parameter.json') as file:
		k = json.load(file)
	return k

def getInJson(parametro1, parametro2):	
	pyson = readParameters()

	for elemento in pyson:
		if parametro1 in elemento:
			if ('sistema' in elemento ) or ('tracker' in elemento )  :
				if elemento['sistema']['computador'] ==  str(socket.gethostname()): # assegura o retorno apenas das informacoes referentes aquela maquina

					return elemento[parametro1][parametro2]
				else:
					continue

			else: # se a informacao nao for sensivel ao ambiente local..
				return elemento[parametro1][parametro2]
	else:
		assert False, 'Nao existe(m) a(s) chave(s) solicitada(s) dentro do arquivo JSON'
