import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


N = 41
sigma = N/6
gaussianFilter =  np.array(signal.gaussian(N,sigma))

print(list(gaussianFilter))
plt.plot(gaussianFilter)
plt.show(gaussianFilter.all())
