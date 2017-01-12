# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 17:40:02 2017

@author: Eric
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import causalMP
from causalMP import snr, plot_spikegram


class SpectroSet:
    
    def __init__(self, data):
        if isinstance(data, str):
            self.load_from_folder(data)
        else:
            self.data = data
            self.ndata = len(data)  
            self.nfreqs = data[0].shape[1]
            
    def load_from_folder(self, folder='../Data/TIMIT/spectrograms'):
        min_length = 26 # TODO: should not be hard-coded
        files = os.listdir(folder)
        file = None
        self.data = []
        for ff in files:
            if ff.endswith('.npy'):
                file = os.path.join(folder,ff)
                spectro = np.load(file)
                if spectro.shape[0] > min_length:
                    self.data.append(spectro)
                    self.nfreqs = spectro.shape[1]
        self.ndata = len(self.data)
        print("Found ", self.ndata, " files")
    
    def rand_stim(self):
        which = np.random.randint(low=0, high=self.ndata)
        signal = self.data[which]
        signal -= signal.mean()
        signal /= signal.std()
        return signal
        
    def show_stim(self, stim, cmap = 'RdBu'):
        stim = -stim # to make red positive
        nfreqs = self.nfreqs
        plt.imshow(stim.T, interpolation= 'nearest',
                   cmap=cmap, aspect='auto', origin='lower')
        plt.ylabel('Frequency')
        plt.xlabel('Time')
        
    def tiled_plot(self, stims, cmap='RdBu'):
        """Tiled plots of the given signals. Zeroth index is which signal.
        Kind of slow, expect about 10s for 100 plots."""
        nstim = stims.shape[0]
        plotrows = int(np.sqrt(nstim))
        plotcols = int(np.ceil(nstim/plotrows))
        f, axes = plt.subplots(plotrows, plotcols, sharex=True, sharey=True)
        for ii in range(nstim):
            axes.flatten()[ii].imshow(-stims[ii].T, interpolation= 'nearest',
                   cmap=cmap, aspect='auto', origin='lower')
        f.subplots_adjust(hspace=0, wspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.setp([a.get_yticklabels() for a in f.axes[:-1]], visible=False)

class SpectroCausalMP(causalMP.CausalMP):
    
    def __init__(self,
                 data = '../Data/TIMIT/spectrograms',
                 nunits = 32,
                 filter_time = 0.200,
                 time_bin = 0.008, 
                 learn_rate = 0.001,
                 thresh = 0.01,
                 normed_thresh = None,
                 max_iter = 1,
                 paramfile = 'spectroCMP.pickle'):
         
        self.nunits = nunits            
        self.thresh = thresh
        self.max_iter = max_iter
        
        self.lfilter = int(filter_time / time_bin)
        self.time_bin = time_bin
        
        self.stims = SpectroSet(data)
        self.nfreqs = self.stims.nfreqs
        self.normed_thresh = normed_thresh or 2/np.sqrt(self.lfilter*self.nfreqs)
        
        self.initialize_graph(learn_rate)
        
        self.errorhist = np.array([])
        self.actfrachist = np.array([])
        self.L0acts = np.zeros(self.nunits)
        self.L1acts = np.zeros(self.nunits)
        self.L2acts = np.zeros(self.nunits)
        self.paramfile = paramfile
        
    
    def initialize_graph(self, learn_rate):
        self.X = tf.placeholder(tf.float32, shape=[None, self.nfreqs], name='signal')
        
        self.filters = tf.Variable(self.initial_filters(), name='filters')
        
        # inference graph
        self.Xnow = tf.placeholder(tf.float32, shape=[self.lfilter, self.nfreqs], name='signal_segment')
        flatfilters = tf.reshape(self.filters, [self.nunits, -1])
        flatsegment = tf.reshape(self.Xnow, [-1,1])
        self.filX = tf.squeeze(tf.matmul(flatfilters, flatsegment))
        self.candidate = tf.cast(tf.argmax(tf.abs(self.filX), axis=0), tf.int32)
        self.cand_contrib = tf.scalar_mul(self.filX[self.candidate],self.filters[self.candidate])
        
        # reconstruction and learning graph
        self.final_coeffs = tf.placeholder(tf.float32, shape=[self.nunits, None], name='final_coefficients')
        rev_filters = tf.reverse(self.filters, dims=[False, True, False])
        trans_coeffs = tf.transpose(self.final_coeffs, [1,0])
        self.Xhat = tf.nn.convolution(tf.expand_dims(trans_coeffs, axis=0), 
                         tf.transpose(rev_filters,[1,0,2]),
                         padding="VALID")
        _, signal_power = tf.nn.moments(self.X, axes=[0])
        self.loss = tf.reduce_mean(tf.square(self.X - tf.squeeze(self.Xhat)) 
                                    / tf.sqrt(signal_power), name='loss')
        self._learn_rate = tf.Variable(learn_rate, trainable=False)
        learner = tf.train.AdamOptimizer(learning_rate=self._learn_rate)
        self.learn_op = learner.minimize(self.loss, var_list=[self.filters])
        normedfilters = tf.nn.l2_normalize(flatfilters, dim=1, epsilon=1e-30)
        self.normalize = self.filters.assign(tf.reshape(normedfilters, [self.nunits, self.lfilter, self.nfreqs]))
        
        # initialize session and variables
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
        config = tf.ConfigProto(gpu_options=gpu_options)
        self.sess = tf.Session(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.normalize)
        
    def initial_filters(self):
        return tf.random_normal([self.nunits, self.lfilter, self.nfreqs])
                
    def infer(self, signal):
        phi = np.reshape(self.phi, [self.nunits, -1])
        resid = np.concatenate([signal, np.zeros((self.lfilter-1,self.nfreqs))],axis=0)
        recon = np.zeros_like(resid)
        lspikes = resid.shape[0]
        spikes = np.zeros([self.nunits, lspikes])
        for tt in range(self.lfilter, lspikes + 1):
            keepgoing = True
            thisiter = 0
            while keepgoing and thisiter < self.max_iter:
                segment = resid[tt-self.lfilter:tt].flatten()
                filX = phi @ segment
                cand = np.argmax(np.abs(filX))
                sp = filX[cand]
                segnorm = np.linalg.norm(segment)
                if np.abs(sp) > self.thresh and np.abs(sp)/segnorm > self.normed_thresh:
                    contrib = np.reshape(sp*phi[cand], [self.lfilter,self.nfreqs])
                    resid[tt-self.lfilter:tt] -= contrib
                    recon[tt-self.lfilter:tt] += contrib
                    spikes[cand, tt-1] = sp
                else:
                    keepgoing = False
                thisiter += 1
        return spikes, recon
    
    def test_inference(self, length = 1000):
        sig = self.stims.rand_stim()[:length]
        length = sig.shape[0]
        spikes, recon = self.infer(sig)
        recon = recon[:length]
        plt.figure()
        plt.subplot(3,1,1)
        self.stims.show_stim(sig)
        plt.title('Original signal')
        plt.subplot(3,1,2)
        self.stims.show_stim(recon)
        plt.title('Reconstruction')
        plt.subplot(3,1,3)
        plot_spikegram(spikes, sample_rate=self.time_bin, markerSize=1)
        print('Signal-noise ratio: ', snr(sig, recon), " dB")    
        
        
    def get_masks(self):
        return None
        
    @property
    def phi(self):
        return self.sess.run(self.filters)    
    @phi.setter
    def phi(self, phi):
        """Also updates masks."""
        self.sess.run(self.filters.assign(phi))