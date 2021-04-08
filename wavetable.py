import numpy as np
import audio as au
import scipy.signal as sig
from numba import jit, njit
from matplotlib import pyplot as plt
import pickle

fs = 44100

## Square

def genSq():
    wave = np.array([au.square(w) for w in np.linspace(-2*np.pi, 2*np.pi, 90000)])
    filSq = np.zeros([8000, 30000])

    for fc in range(1,8000):
        print(fc)
        w = fc / (fs / 2) 
        b, a = sig.butter(5, w, "lowpass")
        initial = sig.lfiltic(b, a, np.zeros(1), np.zeros(1))
        filSq[fc, :] = sig.lfilter(b, a, wave, zi=initial)[0][30000:60000]
        filSq[fc, :] /= np.max(filSq[fc, :])
    
    return filSq

filSq = genSq()

np.save("sqTable.npy", filSq)

filSq = np.load("sqTable.npy")



while 1:
    f = int(input("enter fc: "))
    plt.plot(filSq[f, :])
    plt.show()


