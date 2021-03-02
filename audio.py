
import numpy as np
import sounddevice as sd
import scipy.signal as sig
import time
import mido

t0 = 0
blockTime = 0.01
blockSize = int(44100 * blockTime)

class passFilter:
    def __init__(self, src, fc, fs, mode="low"):
        self.src = src
        self.fc = fc
        self.fs = fs
        self.mode = mode
        w = fc / (fs / 2) 
        self.b, self.a = sig.butter(5, w, mode)
    
    def setMode(self, mode):
        self.mode = mode
        w = self.fc / (self.fs / 2) 
        self.b, self.a = sig.butter(5, w, mode)
        
    def setFc(self, fc):
        self.fc = fc
        w = fc / (self.fs / 2) 
        self.b, self.a = sig.butter(5, w, self.mode)
    
    def nextN(self, n):
        inp = self.src.nextN(n)
        out = sig.filtfilt(self.b, self.a, inp)
        return out

class mixer:
    def __init__(self, src, mix):
        self.mix = mix
        self.src = src
        
    def nextN(self, n):
        samples = [s.nextN(n) for s in self.src]
        mix = sum([s*self.mix[i] for i, s in enumerate(samples)])
        mix = mix/sum(self.mix)
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
 
def noteToFreq(note):
    return (2**((note-69)/12))*440
 
def midi_callback(message):
    if message.type == "note_on":
        for osc in OSCS:
            osc.setF(noteToFreq(message.note))
        env.trig()
    
    if message.type == "note_off":
        env.rel()

def audio_callback(outdata, frames, time, flags):
    outdata[:,0] = fil.nextN(blockSize)
    return None

outStream = sd.OutputStream(samplerate=44100, blocksize=blockSize, channels=1, callback=audio_callback)
sinosc = sineOsc(250, 44100)
squosc = squareOsc(250, 44100)
OSCS = [sinosc,squosc]
mix = mixer(OSCS ,[0.5,0.5])
env = envelope(0.1,0.5,0.2,0.7, mix, 44100)
fil = passFilter(env, 500, 44100)

print(mido.get_input_names())
port = mido.open_input(callback=midi_callback)

with outStream:
    while 1:
        fc = float(input("cutoff: "))
        fil.setFc(fc)
        
        


