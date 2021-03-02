
import numpy as np
import sounddevice as sd
import scipy.signal as sig
import time

t0 = 0
blockTime = 0.1
blockSize = int(44100 * blockTime)

class mixer:
    def __init__(self, src, mix):
        self.mix = mix
        self.src = src
        
    def nextN(self, n):
        samples = [s.nextN(n) for s in self.src]
        mix = sum([s*self.mix[i] for i, s in enumerate(samples)])
        mix = mix/mix.max()
        return mix
        
class envelope:
    def __init__(self, A, D, S, R, src, fs):
        self.A = A
        self.D = D
        self.S = S
        self.R = R 
        self.fs = fs
        self.src = src

        self.triggered = 0
        self.releasing = 0
        self.trigIdx = 0
        self.relIdx = 0
        
        self.ad = np.concatenate([np.linspace(0,1,int(A*fs)),np.linspace(1,S,int(D*fs))])
        self.r = np.linspace(S,0, int(R*fs))
        self.adlen = len(self.ad)
        self.rlen = len(self.r)
    
    def trig(self):
        self.triggered = True
        self.releasing = False
        self.trigIdx = 0
    
    def rel(self):
        self.triggered = False
        self.releasing = True
        self.relIdx = 0
        
        
    def nextN(self, n):
        inp = self.src.nextN(n)
        i=0
        out = np.empty(n)
        while i < n:
            if self.triggered:
                if self.trigIdx < self.adlen:
                    out[i] = inp[i] * self.ad[self.trigIdx]
                    i += 1
                    self.trigIdx += 1
                
                else:
                    out[i] = inp[i] * self.S
                    i+=1
                
            if self.releasing:
                if self.relIdx < self.rlen:
                    out[i] = inp[i] * self.r[self.relIdx]
                    i += 1
                    self.relIdx += 1
                
                else:
                    out[i] = 0
                    i+=1
        
        return out
        

class sineOsc:
    def __init__(self, f, fs):
        self.wave = np.sin(np.linspace(0, 2*np.pi, int(fs/f)))
        self.samples = len(self.wave)
        self.fs = fs
        self.f = f
        self.idx = 0
        self.phase = self.idx / (fs/f)
    
    def setF(self, f):
        self.wave = np.sin(np.linspace(0, 2*np.pi, int(self.fs/f)))
        self.samples = len(self.wave)
        self.f = f
        self.idx = int(self.phase *  (self.fs/self.f))
    
    def nextN(self, n):
        out = []
        for i in range(n):
            out.append(self.wave[(i+self.idx)%self.samples])
        
        self.idx = (self.idx + n)%self.samples
        self.phase = (self.idx / (self.fs/self.f))%(2*np.pi)
        return np.array(out)

class squareOsc(sineOsc):
    def __init__(self, f, fs, duty=0.5):
        self.wave = sig.square(np.linspace(0, 2*np.pi, int(fs/f)), duty)
        self.samples = len(self.wave)
        self.fs = fs
        self.f = f
        self.idx = 0
        self.phase = self.idx / (fs/f)
    
    def setF(self, f):
        self.wave = sig.square(np.linspace(0, 2*np.pi, int(self.fs/f)))
        self.samples = len(self.wave)
        self.f = f
        self.idx = int(self.phase *  (self.fs/self.f))
        
    def setDuty(self, d):
        self.wave = sig.square(np.linspace(0, 2*np.pi, int(fs/f)), d)

class sawOsc(sineOsc):
    def __init__(self, f, fs, width=1):
        self.wave = sig.sawtooth(np.linspace(0, 2*np.pi, int(fs/f)), width)
        self.samples = len(self.wave)
        self.fs = fs
        self.f = f
        self.idx = 0
        self.phase = self.idx / (fs/f)
        
    def setF(self, f):
        self.wave = sig.sawtooth(np.linspace(0, 2*np.pi, int(self.fs/f)))
        self.samples = len(self.wave)
        self.f = f
        self.idx = int(self.phase *  (self.fs/self.f))
    
    def setWidth(self, w):
        self.wave = sig.sawtooth(np.linspace(0, 2*np.pi, int(fs/f)), w)
     

def callback(outdata, frames, time, flags):
    global t0
    global mix
    
    t = time.outputBufferDacTime
    if not t0:
        t0 = t
    t = t - t0
    
    outdata[:,0] = env.nextN(blockSize)
    return None

outStream = sd.OutputStream(samplerate=44100, blocksize=blockSize, channels=1, callback=callback)
mix = mixer([sineOsc(125, 44100), squareOsc(500, 44100)],[0.5,0.5])
env = envelope(0.1,0.5,0.2,0.7, mix, 44100)
f = 500
with outStream:
    while 1:
        i = input("t or r: ")
        if i == "t":
            env.trig()
        if i == "r":
            env.rel()
        
        


