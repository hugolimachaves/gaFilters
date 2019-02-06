import pickle
import numpy as np

pickle_off = open("filtro_1.pickle","rb")
emp = pickle.load(pickle_off)
print(emp)