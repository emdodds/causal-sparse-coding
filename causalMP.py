# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 14:59:37 2017

@author: Eric
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.io import wavfile
from scipy import signal as scisig
import pickle

def snr(signal, recon):
    """Returns signal-noise ratio in dB."""
    ratio = np.var(signal)/np.var(signal-recon)
    return 10*np.log10(ratio)
    
# dynamic compressive gammachirp
def dcGC(t,f):
    """Dynamic compressive gammachirp filter as defined by Irino,
    with parameters from Park as used in Charles, Kressner, & Rozell.
    The log term is regularized to log(t + 0.00001).
    t : time in seconds, greater than 0
    f : characteristic frequency in Hz
    One but not both arguments may be numpy arrays.
    """
    ERB = 0.1039*f + 24.7
    return t**3 * np.exp(-2*np.pi*1.14*ERB*t) * np.cos(2*np.pi*f*t + 0.979*np.log(t+0.000001))

# adapted from scipy cookbook
lowcut = 100
highcut = 6000
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = scisig.butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scisig.lfilter(b, a, data)
    return y
    
def plot_spikegram( spikes, sample_rate, markerSize = .0001 ):
    """adapted from https://github.com/craffel/spikegram-coding/blob/master/plotSpikeGram.py"""
    nkernels = spikes.shape[0]
    indices = np.transpose(np.nonzero(spikes))
    print(indices.shape)
    scalesKernelsAndOffsets = [(spikes[idx[0],idx[1]], idx[0], idx[1]) for idx in indices]
    
    for scale, kernel, offset in scalesKernelsAndOffsets:
        # Put a dot at each spike location.  Kernels on y axis.  Dot size corresponds to scale
        plt.plot( offset/sample_rate, nkernels-kernel, 'k.', 
                 markersize=markerSize*np.abs( scale ) )
    plt.title( "Spikegram" )
    plt.xlabel( "Time (s)" )
    plt.ylabel( "Kernel" )
    plt.axis( [0.0, spikes.shape[1]/sample_rate, 0.0, nkernels] )
    plt.show()
    
class SignalSet:
    
    def __init__(self, sample_rate = 16000, data = '../Data/TIMIT/'):
        self.sample_rate = sample_rate
        if isinstance(data, str):
            self.load_from_folder(data)
        else:
            self.data = data
            self.ndata = len(data)            
            
    def load_from_folder(self, folder = '../Data/TIMIT/'):
        min_length = 800 # TODO: should not be hard-coded
        files = os.listdir(folder)
        file = None
        self.data = []
        for ff in files:
            if ff.endswith('.wav'):
                file = os.path.join(folder,ff)
                rate, signal = wavfile.read(file)
                if rate != self.sample_rate:
                    raise NotImplementedError('The signal in ' + ff +
                    ' does not match the given sample rate.')
                if signal.shape[0] > min_length:
                    # bandpass
                    signal = signal/signal.std()
                    signal = butter_bandpass_filter(signal, lowcut, highcut,
                                                    self.sample_rate, order=5)
                    self.data.append(signal)
        self.ndata = len(self.data)
        print("Found ", self.ndata, " files")
        
    def rand_stim(self):
        """Get one random signal."""
        which = np.random.randint(low=0, high=self.ndata)
        signal = self.data[which]
        signal /= np.max(signal) # as in Smith & Lewicki
        return signal

class CausalMP:
    
    def __init__(self,
                 data = '../Data/TIMIT/',
                 nunits = 32,
                 filter_time = 0.05,
                 learn_rate = 0.01,
                 thresh = 1.5,
                 tf_inference = False):
        
        self.thresh = 0.6
        self.tf_inference = tf_inference
        self.sample_rate = 16000
        self.nunits = 32
        self.lfilter = int(filter_time * self.sample_rate)
        
        self.initialize_graph(learn_rate)
        self.stims = self.load_data(data)
        
        self.errorhist = np.array([])
        self.actfrachist = np.array([])
        self.paramfile= 'causalMP.pickle'
    
    def load_data(self, data):
        return SignalSet(self.sample_rate, data)
    
    def initialize_graph(self, learn_rate):
        self.X = tf.placeholder(tf.float32, shape=[None], name='signal')
        
        self.filters = tf.Variable(self.initial_filters(), name='filters')
        
        # inference graph
        self.Xnow = tf.placeholder(tf.float32, shape=[self.lfilter], name='signal_segment')
        self.filX = tf.squeeze(tf.matmul(self.filters, tf.expand_dims(self.Xnow,dim=1)))
        self.candidate = tf.cast(tf.argmax(tf.abs(self.filX), axis=0), tf.int32)
        self.cand_contrib = tf.scalar_mul(self.filX[self.candidate],self.filters[self.candidate])
        
        # learning graph
        self.final_coeffs = tf.placeholder(tf.float32, shape=[self.nunits, None], name='final_coefficients')
        self.rev_filters = tf.reverse(self.filters, dims=[False, True])
        self.Xhat = tf.nn.convolution(tf.expand_dims(self.final_coeffs, dim=0), 
                         tf.transpose(tf.expand_dims(self.rev_filters, dim=2),[1,0,2]),
                         padding="VALID", data_format="NCW")
        _, signal_power = tf.nn.moments(self.X, axes=[0])
        self.loss = tf.reduce_mean(tf.square(self.X - tf.squeeze(self.Xhat)) 
                                    / tf.sqrt(signal_power), name='loss')
        self._learn_rate = tf.Variable(learn_rate, trainable=False)
        learner = tf.train.AdamOptimizer(learning_rate=self._learn_rate)
        self.learn_op = learner.minimize(self.loss, var_list=[self.filters])
        self.normalize = self.filters.assign(tf.nn.l2_normalize(self.filters, dim=1, epsilon=1e-30))
        
        # initialize session and variables
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.normalize)
    
    def initial_filters(self, gammachirp=False):
        if gammachirp:
            gammachirps = np.zeros([self.nunits, self.lfilter])
            freqs = np.logspace(np.log10(100), np.log10(6000), self.nunits)
            times = np.linspace(0,self.lfilter/self.sample_rate,self.lfilter)
            for ii in range(self.nunits):
                gammachirps[ii] = dcGC(times, freqs[ii])
            return gammachirps        
        else:
            return tf.random_normal([self.nunits, self.lfilter])
    
    def infer(self, signal):
        if self.tf_inference:
            return self.tf_infer(signal)
        else:
            return self.np_infer(signal)
            
    def tf_infer(self, signal):
        """Not guaranteed to work or be faster than numpy version."""
        resid = np.concatenate([signal, np.zeros(self.lfilter-1)], axis=0)
        recon = np.zeros_like(resid)
        lspikes = resid.shape[0]
        spikes = np.zeros([self.nunits, lspikes])
        for tt in range(self.lfilter, lspikes+1):
            filout, cand, contrib = sess.run((self.filX, self.candidate, self.cand_contrib),
                                             feed_dict = {Xnow : resid[tt-self.lfilter:tt]})
            sp = filout[cand]
            if np.abs(sp) > self.thresh:
                resid[tt-self.lfilter:tt] -= contrib
                recon[tt-self.lfilter:tt] += contrib
                spikes[cand, tt-1] = sp
        return spikes, recon
        
    def np_infer(self, signal):
        """Returns array of spikes and a reconstruction of the given signal.
        The reconstruction is self.lfilter-1 longer than the signal."""
        phi = self.phi
        resid = np.concatenate([signal, np.zeros(self.lfilter-1)], axis=0)
        recon = np.zeros_like(resid)
        lspikes = resid.shape[0]
        spikes = np.zeros([self.nunits, lspikes])
        for tt in range(self.lfilter, lspikes+1):
            filX = phi @ resid[tt-self.lfilter:tt]
            cand = np.argmax(np.abs(filX))
            sp = filX[cand]
            if np.abs(sp) > self.thresh:
                contrib = filX[cand]*phi[cand]
                resid[tt-self.lfilter:tt] -= contrib
                recon[tt-self.lfilter:tt] += contrib
                spikes[cand, tt-1] = sp
        return spikes, recon
        
    def test_inference(self, length=10000):
        sig = self.stims.rand_stim()[:length]
        spikes, recon = self.infer(sig)
        recon = recon[:length]
        times = np.arange(length) / self.sample_rate
        plt.figure()
        plt.subplot(3,1,1)
        plt.plot(times, sig)
        plt.title('Original signal')
        plt.subplot(3,1,2)
        plt.plot(times, recon)
        plt.title('Reconstruction')
        plt.subplot(3,1,3)
        plot_spikegram(spikes, sample_rate=self.sample_rate, markerSize=1)
        print('Signal-noise ratio: ', snr(sig, recon))
        
    
    def learn_step(self):
        signal = self.stims.rand_stim()
        lsignal = signal.shape[0]
        spikes, recon = self.infer(signal)
        feed_dict = {self.X : signal, self.final_coeffs : spikes}
        error, _ = self.sess.run((self.loss, self.learn_op), feed_dict = feed_dict)
        self.sess.run(self.normalize)
        nspikes = np.count_nonzero(spikes)
        act_fraction = nspikes/lsignal
        return error, act_fraction
    
    def train(self, ntrials=10000):
        # TODO: add history accounting and other bookkeeping
        for ii in range(ntrials):
            error, act_frac = self.learn_step()
            self.errorhist = np.append(self.errorhist, error)
            self.actfrachist = np.append(self.actfrachist, act_frac)        

    def save(self, filename=None):
        if filename is None:
            filename = self.paramfile
        with open(filename,'wb') as ff:
            pickle.dump([self.phi, self.params, self.histories], f)
        self.paramfile = filename
        
    def load(self, filename):
        with open(filename, 'rb') as ff:
            self.phi, self.params, self.histories = pickle.load(ff)
        self.paramfile = filename
    
    @property
    def phi(self):
        return self.sess.run(self.filters)
    @phi.setter
    def phi(self, phi):
        self.sess.run(self.filters.assign(phi))
    
    @property
    def learn_rate(self):
        return self.sess.run(self._learn_rate)        
    @learn_rate.setter
    def learn_rate(self, new_rate):
        self.sess.run(self._learn_rate.assign(new_rate))
        
    @property
    def params(self):
        return (self.thresh, self.learn_rate)
    @params.setter
    def params(self, params):
        self.thresh, self.learn_rate = params
        
    @property
    def histories(self):
        return (self.errorhist, self.actfrachist)
     
    @histories.setter
    def histories(self, histories):
        self.errorhist, self.actfrachist = histories