
import numpy as np
import sounddevice as sd
import scipy.signal as sig
import time
import mido
from matplotlib import pyplot as plt

class attenuverter:
    def __init__(self, src, fac):
        self.src = src
        self.fac = fac
        
    def nextN(self, n):
        return np.clip(self.src.nextN(n) * self.fac, -0.999, 0.999)

class distortion:
    def __init__(self, src, amt):
        self.amt = amt
        self.factor = 1/amt
        self.src = src
        self._type = "fx"
    
    def setAmt(self, amt):
        self.amt = amt
        self.factor = 1/amt
        
    def nextN(self, n):
        inp = self.src.nextN(n)
        signs = np.sign(inp)
        return (abs(inp) ** self.factor) * signs
        
class passFilter:
    def __init__(self, src, fc, fs, mode="lowpass"):
        self.src = src
        self.fc = fc
        self.fs = fs
        self.mode = mode
        w = fc / (fs / 2) 
        self.b, self.a = sig.butter(5, w, mode)
        
        self.initial = sig.lfiltic(self.b, self.a, np.zeros(1), np.zeros(1))
        self._type = "fil"
    
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
        out, zf = sig.lfilter(self.b, self.a, inp, zi=self.initial)
        self.initial = zf
        
        return out

class mixer:
    def __init__(self, src, mix):
        self.mix = mix
        self.src = src
        self._type = "mix"
        
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
        
        self._type = "env"
    
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
        out = np.zeros(n)
        while i < n:
            if self.triggered:
                if self.trigIdx < self.adlen:
                    out[i] = inp[i] * self.ad[self.trigIdx]
                    self.currentLevel = self.ad[self.trigIdx]
                    self.trigIdx += 1
                
                else:
                    out[i] = inp[i] * self.S
                       
            if self.releasing:
                if self.relIdx < self.rlen:
                    out[i] = inp[i] * self.r[self.relIdx] * (self.currentLevel/self.S)
                    self.relIdx += 1
                
                else:
                    out[i] = 0
                    
            i += 1
            
        return out
        

class sineOsc:
    def __init__(self, f, fs):
        self.wave = np.sin(np.linspace(0, 2*np.pi, int(fs/f)))
        self.samples = len(self.wave)
        self.fs = fs
        self.f = f
        self.idx = 0
        self.phase = (self.idx / self.samples) * 2 * np.pi
        self._type = "osc"
    
    def setF(self, f):
        self.wave = np.sin(np.linspace(0, 2*np.pi, int(self.fs/f)))
        self.samples = len(self.wave)
        self.idx = int((self.phase/2*np.pi) *  self.samples)
        self.f = f
        
    
    def nextN(self, n):
        out = []
        for i in range(n):
            out.append(self.wave[self.idx])
            self.idx += 1
            self.idx = self.idx % self.samples
            print(self.idx)
        
        self.phase = (self.idx / self.samples) * 2 * np.pi
        return np.array(out)

class squareOsc(sineOsc):
    def __init__(self, f, fs, duty=0.5):
        self.wave = sig.square(np.linspace(0, 2*np.pi, int(fs/f)), duty)
        self.samples = len(self.wave)
        self.fs = fs
        self.f = f
        self.idx = 0
        self.phase = self.idx / (fs/f)
        self._type = "osc"
    
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
        self._type = "osc"
        
    def setF(self, f):
        self.wave = sig.sawtooth(np.linspace(0, 2*np.pi, int(self.fs/f)))
        self.samples = len(self.wave)
        self.f = f
        self.idx = int(self.phase *  (self.fs/self.f))
    
    def setWidth(self, w):
        self.wave = sig.sawtooth(np.linspace(0, 2*np.pi, int(fs/f)), w)

class synth:
    def __init__(self, fs, blockTime):
        self.fs = fs
        self.blockTime = blockTime
        self.blockSize = int(fs * blockTime)
        self.oscs = []
        self.envs = []
        self.modules = []
        self.output = None
        self.audioStream = sd.OutputStream(samplerate=fs, blocksize=self.blockSize, channels=1, callback=self.audio_callback)
        self.midiPort = mido.open_input(callback=self.midi_callback)
        self.wave = np.zeros(self.blockSize*2)
    
    def audio_callback(self, outdata, frames, time, flags):
        if self.output:
            o = self.output.nextN(self.blockSize)
            outdata[:,0] = o - np.mean(o)
            self.wave[self.blockSize:] = self.wave[:-self.blockSize]
            self.wave[:self.blockSize] = o
             
        return None
        
    def midi_callback(self, message):
        if message.type == "note_on":
            for osc in self.oscs:
                osc.setF(noteToFreq(message.note))
                
            for env in self.envs:
                env.trig()
    
        if message.type == "note_off":
            for env in self.envs:
                env.rel()
    
    def setModulesSeries(self, modules):
        self.modules = modules
        self.envs = []
        self.oscs = []
        for module in modules:
            if module._type == "env":
                self.envs.append(module)
            elif module._type == "osc":
                self.oscs.append(module)
        
        self.output = modules[-1]
    
    def run(self):
        print("Initialising synth...")
        with self.audioStream:
            print("Synth ready!")
            while 1:
                plt.cla()
                plt.plot(self.wave)
                plt.show()
                plt.pause(0.001)


def noteToFreq(note):
    return (2**((note-69)/12))*440
 




plt.ion()
sinosc = sineOsc(250, 44100)
squosc = squareOsc(250, 44100)
OSCS = [squosc]
mix = mixer(OSCS ,[0.5,0.5])
env = envelope(0.1,0.5,0.2,0.7, sinosc, 44100)
dist = distortion(env, 5)
fil = passFilter(dist, 10000, 44100)


syn = synth(44100, 0.05)
syn.setModulesSeries([sinosc, env])
syn.run()

        
        


