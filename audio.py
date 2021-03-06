
import numpy as np
import sounddevice as sd
import scipy.signal as sig
import time
import mido
from matplotlib import pyplot as plt
import cProfile, pstats
from linetimer import CodeTimer
import pickle
from numba import njit, jit

class attenuverter:
    def __init__(self, fac, src=None):
        self.src = src
        self.fac = fac
        
    def nextN(self, n):
        return np.clip(self.src.nextN(n) * self.fac, -0.999, 0.999)

class distortion:
    def __init__(self, amt, src=None):
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


def wToN(w):
    x = w/2
    LHS = 0.707*(x - (x**3)/6 + (x**5)/120)# - (x**7)/5040
    N = 1
    while True:
        xn = x*N % (2*np.pi)
        RHS = (xn - (xn**3)/6 + (xn**5)/120)/N# - (xn**7)/5040)/N
        if RHS <= LHS:
            return N 
        else:
            N += 1


def fastAverage(array):
    l = len(array)
    skip = (l // 300) + 1
    return sum(array[::skip])/(l/skip)
    
    
class tableSynth:
    def __init__(self, fs, wave="square"):
        if wave == "square":
            self.table = np.load("sqTable.npy", mmap_mode="r")
        
        self.f = 440
        self.fs = fs
        self.phase = 0   # goes 0 to 30k
        self.dp = int((30000 / self.fs) /  (1/self.f))
        self._type = "osc"
        self.blockId = 1
        self.params = {"cutoff": 1}
    
    def addModulator(self, param, modulator):
        self.params[param] = modulator
    
    def setF(self, f):
        self.f = f
        self.dp = int((30000 / self.fs) /  (1/self.f))
        #self.dp = int(np.pi*2 / (self.fs/f))
        
    def getModulations(self, n):
        out = {}
        for key in self.params.keys():
            param = self.params[key]
            if type(param) == modulator:   
                out[key] = param.nextN(n, self.blockId)
            else:
                out[key] = np.array([param]*n)
        
        self.blockId = (self.blockId + 1) % 100        
        return out
        
    
    def scaleCutoff(self, modulation):
        #print(modulation)
        return int((((modulation+1)/2) * 7997) + 1)
        
    
    def nextN(self, n):
        mods = self.getModulations(n)
        out = np.zeros(n)
        for i in range(n):
            self.fc = self.scaleCutoff(mods["cutoff"][i])
            #print(self.fc)
            #print(self.fc)
            self.phase = (self.phase + self.dp) % 30000
            #print(self.fc, self.phase)
            out[i] = self.table[self.fc, self.phase]
        
        return out
                
      
class passFilter:
    def __init__(self, fc, fs, src=None, mode="lowpass"):
        self.src = src
        self.fc = fc
        self.fs = fs
        self.mode = mode
        w = fc / (fs / 2)
        self.N = wToN(w)
        self.memory = []
        self.params = {"cutoff": fc}
        self._type = "fil"
        self.blockId = 1
        with open("FilterLookup.dat", "rb") as f:
            self.lookup = pickle.load(f)
    
    def setMode(self, mode):
        self.mode = mode
        w = self.fc / (self.fs / 2) 
        self.b, self.a = sig.butter(5, w, mode)
    
    def addModulator(self, param, modulator):
        self.params[param] = modulator
    
    
    def getModulations(self, n):
        out = {}
        for key in self.params.keys():
            param = self.params[key]
            if type(param) == modulator:   
                out[key] = param.nextN(n, self.blockId)
            else:
                out[key] = np.array([param]*n)
        
        self.blockId = (self.blockId + 1) % 100        
        return out
    
    
    def setFc(self, fc):
        self.fc = fc
        w = fc / (self.fs / 2) 
        self.b, self.a = sig.butter(5, w, self.mode)
        
    
    def scaleFcMod(self, mod):
        return ((mod+1)/2)[100]
    
    # 
    # def nextN(self, n):
        # mods = self.getModulations(n)
        # for i in range(n):
            # self.setFc(self.scaleFcMod(mods["cutoff"][i]))
            
    
    # 
    # def nextN(self, n):
        # mods = self.getModulations(n)
        # inp = self.src.nextN(n)
        # out = []
        
        # inp = np.pad(inp, 50, mode="edge")
        # fft = np.fft.rfft(inp)
        # print(len(fft))
        # fft = fft[:int(self.scaleFcMod(mods["cutoff"])*len(fft))+200]
        # print(len(fft))
        # out = np.fft.irfft(fft,n=n)
        
        # return out
    # def scaleFcMod(self, mod):
        # return 20 + ((mod+1)/2) * 1e3
    
    # def nextN(self, n):
        # mods = self.getModulations(n)
        # inp = self.src.nextN(n)
        # out = []
        # for i, sample in enumerate(inp):
            # fc = self.scaleFcMod(mods["cutoff"][i])
            # w = 2*np.pi*(fc / self.fs)
            # self.N = self.lookup[round(w)]
            # #print(self.N)
            # while len(self.memory) > self.N:
                # del self.memory[0]
            # self.memory.append(sample)
            # outS = fastAverage(self.memory)
            # out.append(outS)
            
        # return np.array(out)

class mixer:
    def __init__(self, mix, src=None):
        self.mix = mix
        self.src = src
        self._type = "mix"
    
    
    def nextN(self, n):
        samples = [s.nextN(n) for s in self.src]
        mix = sum([s*self.mix[i] for i, s in enumerate(samples)])
        mix = mix/sum(self.mix)
        return mix
        
class envelope:
    def __init__(self, A, D, S, R, fs, src=None, retrig=False):
        self.A = A
        self.D = D
        self.S = S
        self.R = R 
        self.fs = fs
        self.src = src
        self.retrig = retrig

        self.triggered = False
        self.releasing = False
        self.trigIdx = 0
        self.relIdx = 0
        self.currentLevel = 0
        
        self.ad = np.concatenate([np.linspace(0,1,int(A*fs)),np.linspace(1,S,int(D*fs))])
        self.r = np.linspace(S,0, int(R*fs))
        self.adlen = len(self.ad)
        self.rlen = len(self.r)
        
        #w = 10000 / (fs / 2) 
        #self.b, self.a = sig.butter(5, w, "lowpass")
        #self.initial = sig.lfiltic(self.b, self.a, np.zeros(1), np.zeros(1))
        
        self._type = "env"
    
    def trig(self):
        self.triggered = True
        self.releasing = False
        if (self.trigIdx >= self.adlen and self.relIdx >= self.rlen) or self.retrig:
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
            if self.triggered :
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
        
        #out, zf = sig.lfilter(self.b, self.a, out, zi=self.initial)
        #self.initial = zf
        
        return out


class osc:
    def __init__(self, fs, waveFcn, f=440):
        self.waveFcn = waveFcn.waveFcn
        self.params = waveFcn.params
        self.f = f
        self.fs = fs
        self.dp = (2*np.pi) / (fs/f)
        self.phase = 0.0
        self._type = "osc"
        self.blockId = 1
    
    def setF(self, f):
        self.f = f
        self.dp = (2*np.pi) / (self.fs/f)
    
    def addModulator(self, param, modulator):
        self.params[param] = modulator
    
    def getModulations(self, n):
        out = {}
        for key in self.params.keys():
            param = self.params[key]
            if type(param) == modulator:   
                out[key] = param.nextN(n, self.blockId)
            else:
                out[key] = np.array([param]*n)
        
        self.blockId = (self.blockId + 1) % 100        
        return out
        
         
    def nextN(self, n):
        out = []
        mods = self.getModulations(n)
        for i in range(n):
            dc = {k: v[i] for k, v in mods.items()}
            sample = self.waveFcn(self.phase, **dc)
            #print(sample)
            out.append(sample) #check this
            self.phase += self.dp
        
        return np.array(out)    

class waveFcn:
    def __init__(self, wave, params):
        self.waveFcn = wave   # a function f(phase) = value, period 2*pi
        self.params = params


def square(x, duty=0.5):
    x = x%(2*np.pi)
    duty *= 2*np.pi
    g = 50
    
    f1 = g*x
    if -1 < f1 < 1:
        return f1
        
    f2 = -(g*(x-duty))
    if -1 < f2 < 1:
        return f2
            
    f3 = g*(x-(2*np.pi))
    if -1 < f3 < 1:
        return f3
    
    if x <= duty:
        return 1.0
    if x> duty:
        return -1.0

        
def sawtri(x, shape=0.5):
    x = x%(2*np.pi)
    shape *= np.pi/2
    
    f1 = x/(np.pi-shape)
    if -1 < f1 < 1:
        return f1
    
    f2 = (-2*(x-np.pi))/(2*shape + 1e-9)
    if -1 < f2 < 1:
        return f2
    
    f3 = (x-(2*np.pi))/(np.pi-shape)
    if -1 < f3 < 1:
        return f3


SINE = waveFcn(np.sin, {})
SAWTRI = waveFcn(sawtri, {"shape": 0.0})
SQUARE = waveFcn(square, {"duty": 0.5})

def eye(x):
    return x

def sineToPos(y):
    return (y+1)/2
        
class modulator:
    def __init__(self, obj, args, scaleFcn=None):
        self.obj = obj(*args)
        self.cache = None
        self.BlockId = None
        if scaleFcn:
            self.scaleFcn = scaleFcn
        else:
            self.scaleFcn = eye
            
       
    def nextN(self, n, blockId):
        if blockId != self.BlockId:
            self.cache = self.scaleFcn(self.obj.nextN(n))
            
        return self.cache

class synth:
    def __init__(self, fs):
        self.fs = fs
        self.oscs = []
        self.envs = []
        self.modules = []
        self.output = None
        self.audioStream = sd.OutputStream(samplerate=fs, blocksize=0, latency="low", channels=1, callback=self.audio_callback)
        self.midiPort = mido.open_input(callback=self.midi_callback)
        self.wave = np.zeros(2000)
    
    def audio_callback(self, outdata, frames, time, flags):
        if self.output:
            o = self.output.nextN(frames)
            outdata[:,0] = o
            self.wave[:-frames] = self.wave[frames:]
            self.wave[-frames:] = o
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
        
        while self.midiPort.poll():
            pass
    
    def setModulesSeries(self, modules):
        self.modules = modules
        self.envs = []
        self.oscs = []
        lastModule = None
        for module in modules:
            if module._type == "env":
                self.envs.append(module)
                module.src = lastModule
            elif module._type == "osc":
                self.oscs.append(module)
            else:
                module.src = lastModule
            
            lastModule = module
        
        self.output = modules[-1]
    
    def run(self, scope=False):
        with self.audioStream:
            print(f"Synth ready!\nListenting on MIDI port: {self.midiPort.name}")
            
            while 1:
                
                if scope:
                    plt.cla()
                    plt.plot(self.wave)
                    plt.grid(True)
                    plt.ylim([-1,1])
                    plt.show()
                    plt.pause(0.001)
                else:
                    pass


def noteToFreq(note):
    return (2**((note-69)/12))*440

if __name__ == "__main__":
    F_S = 44100
    plt.ion()
    
    print("Initialising synth...")
    #sinosc = osc(F_S, SINE)
    mod = modulator(osc, [F_S, SINE, 1])
    osc = tableSynth(F_S)
    osc.addModulator("cutoff", mod)
    env = envelope(0.1, 0.3, 0.7, 0.7, F_S)
    
    syn = synth(F_S)
    syn.setModulesSeries([osc, env])
    
    ## play a file
    # with syn.audioStream:
        # for msg in mido.MidiFile('midi2.mid').play():
            # syn.midi_callback(msg)
            
    syn.run(True)


        


