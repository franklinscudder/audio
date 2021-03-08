
import numpy as np
import sounddevice as sd
import scipy.signal as sig
import time
import mido
from matplotlib import pyplot as plt
import cProfile, pstats



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
        
class passFilter:
    def __init__(self, fc, fs, src=None, mode="lowpass"):
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
        
        w = 10000 / (fs / 2) 
        self.b, self.a = sig.butter(5, w, "lowpass")
        self.initial = sig.lfiltic(self.b, self.a, np.zeros(1), np.zeros(1))
        
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
        
        out, zf = sig.lfilter(self.b, self.a, out, zi=self.initial)
        self.initial = zf
        
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
            sample = self.waveFcn(self.phase, **{k: v[i] for k, v in mods.items()})
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
    elif -1 < f2 < 1:
        return f2
            
    f3 = g*(x-(2*np.pi))
    elif -1 < f3 < 1:
        return f3
    
    elif x <= duty:
        return 1.0
    elif x> duty:
        return -1.0
        
def sawtri(x, shape=0.5):
    x = x%(2*np.pi)
    shape *= np.pi/2
    
    f1 = x/(np.pi-shape)
    if -1 < f1 < 1:
        return f1
    
    f2 = (-2*(x-np.pi))/(2*shape + 1e-9)
    elif -1 < f2 < 1:
        return f2
    
    f3 = (x-(2*np.pi))/(np.pi-shape)
    elif -1 < f3 < 1:
        return f3


SINE = waveFcn(np.sin, {})
SAWTRI = waveFcn(sawtri, {"shape": 0.0})
SQUARE = waveFcn(square, {"duty": 0.5})

def sineToPos(y):
    return (y+1)/2
        
class modulator:
    def __init__(self, obj, args, scaleFcn):
        self.obj = obj(*args)
        self.cache = None
        self.BlockId = None
        self.scaleFcn = scaleFcn
        
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
 
F_S = 44100
plt.ion()
sinosc = osc(F_S, SINE)
mod = modulator(osc, [F_S, SINE, 1], sineToPos)
squosc = osc(F_S, SQUARE)
squosc.addModulator('duty', mod)
sawosc = osc(F_S, SAWTRI)
# mix = mixer([0.5,0.5])
env = envelope(.1,1,0.7,.3, F_S)
# dist = distortion(5)
# fil = passFilter(10000, F_S)

print("Initialising synth...")
syn = synth(F_S)
syn.setModulesSeries([sawosc, env])

syn.run(True)


        


