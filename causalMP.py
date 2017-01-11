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
        
    def write_sound(self, filename, signal):
        wavfile.write(filename, self.sample_rate, signal)
        
    def tiled_plot(self, stims):
        """Tiled plots of the given signals. Zeroth index is which signal.
        Kind of slow, expect about 10s for 100 plots."""
        nstim = stims.shape[0]
        plotrows = int(np.sqrt(nstim))
        plotcols = int(np.ceil(nstim/plotrows))
        f, axes = plt.subplots(plotrows, plotcols, sharex=True, sharey=True)
        for ii in range(nstim):
            axes.flatten()[ii].plot(stims[ii])
        f.subplots_adjust(hspace=0, wspace=0)
        plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        plt.setp([a.get_yticklabels() for a in f.axes[:-1]], visible=False)

class CausalMP:
    
    def __init__(self,
                 data = '../Data/TIMIT/',
                 nunits = 32,
                 filter_time = 0.05,
                 learn_rate = 0.01,
                 thresh = 0.5,
                 normed_thresh = 0.1,
                 max_iter = 1,
                 mask_epsilon = None,
                 tf_inference = False,
                 paramfile= 'causalMP.pickle'):
        """
        Causal Matching Pursuit tries at each time to add the coefficient(s)
        that most improve(s) a linear representation of a given signal,
        using only the portion of the signal before that time.

        Parameters:
        ---------------------------------------------------------------------
        nunits : (int) number of filters used
        filter_time : (float) length of filters in seconds
        learn_rate : (float) rate used for dictionary learning
        thresh : (float) coefficients below this value are rejected
        normed_thresh : (bool) coefficients are rejected if below this value
            after division by signal segment norm
        max_iter : (int) maximum number of coefficients per time point
        mask_epsilon : (float) small number for deciding region of filter close
            enough to zero to ignore for normed_thresh. Default determined by
            filter length.
        tf_inference : (bool) use tensorflow inference method if True
        paramfile : (str) filename where parameters, dict, histories are saved
        """
        
        
        self.thresh = 0.6
        self.normed_thresh = normed_thresh
        self.tf_inference = tf_inference
        self.sample_rate = 16000
        self.nunits = 32
        self.lfilter = int(filter_time * self.sample_rate)
        self.mask_epsilon = mask_epsilon or 0.01*np.sqrt(1/self.lfilter)
        self.max_iter = max_iter
        
        self.initialize_graph(learn_rate)
        self.masks = self.get_masks()
        self.stims = self.load_data(data)
        
        self.errorhist = np.array([])
        self.actfrachist = np.array([])
        self.L0acts = np.zeros(self.nunits)
        self.L1acts = np.zeros(self.nunits)
        self.L2acts = np.zeros(self.nunits)
        self.paramfile = paramfile
    
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
       
    def get_masks(self):
        """Returns an array where each row is a binary mask for a filter. The
        mask should be zero outside the region where the filter has significant
        support. Used for ignoring part of signal segments when assessing
        whether a candidate coefficient clears the normed_thresh.
        For now this method assumes that the correct mask has zeros to the left
        and ones to the right of some midpoint. This is consistent with filters
        learned in early experiments."""
        masks = self.phi > self.mask_epsilon
        starts = np.argmax(masks, axis=1) # returns *first* maximum
        for ind in range(masks.shape[0]):
            masks[ind,starts[ind]:] = 1
        return masks
        
    # Inference
    ##########################################################################
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
            filout, cand, contrib = self.sess.run((self.filX, self.candidate, self.cand_contrib),
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
            keepgoing = True
            thisiter = 0
            while keepgoing and thisiter < self.max_iter:
                segment = resid[tt-self.lfilter:tt]
                filX = phi @ segment
                cand = np.argmax(np.abs(filX))
                sp = filX[cand]
                segnorm = np.linalg.norm(segment[self.masks[cand]])
                if np.abs(sp) > self.thresh and np.abs(sp)/segnorm > self.normed_thresh:
                    if spikes[cand, tt-1] == 0:
                        contrib = filX[cand]*phi[cand]
                        resid[tt-self.lfilter:tt] -= contrib
                        recon[tt-self.lfilter:tt] += contrib
                        spikes[cand, tt-1] = sp
                    else:
                        keepgoing = False
                else:
                    keepgoing = False
                thisiter += 1
                
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
        print('Signal-noise ratio: ', snr(sig, recon), " dB")
        
    # Learning
    ##########################################################################
    def learn_step(self):
        signal = self.stims.rand_stim()
        lsignal = signal.shape[0]
        spikes, recon = self.infer(signal)
        feed_dict = {self.X : signal, self.final_coeffs : spikes}
        error, _ = self.sess.run((self.loss, self.learn_op), feed_dict = feed_dict)
        self.sess.run(self.normalize)
        self.masks = self.get_masks()
        # record and return stats
        nspikes = np.count_nonzero(spikes)
        act_fraction = nspikes/lsignal
        eta = 0.01
        self.L0acts = (1-eta)*self.L0acts + eta*np.mean(spikes != 0, axis=1)
        self.L1acts = (1-eta)*self.L1acts + eta*np.mean(np.abs(spikes), axis=1)
        self.L2acts = (1-eta)*self.L2acts + eta*np.mean(spikes**2, axis=1)
        return error, act_fraction
    
    def train(self, ntrials=10000):
        for ii in range(ntrials):
            error, act_frac = self.learn_step()
            self.errorhist = np.append(self.errorhist, error)
            self.actfrachist = np.append(self.actfrachist, act_frac)      
            if ii % 50 == 0:
                print(ii)
                if ii % 1000 == 0 and ii != 0:
                    self.save()
          
          
    # Visualization
    ##########################################################################
    def show_dict(self):
        self.stims.tiled_plot(self.phi)
        
    def show_spectra(self):
        """Show a tiled plot of the power spectra of the current dictionary."""
        spectra = np.square(np.abs(np.fft.rfft(self.phi, axis=1)))
        self.stims.tiled_plot(spectra)
        
    def progress_plot(self, window_size=1000, norm=1, start=0, end=-1):
        """Plots a moving average of the error and activity history with the
        given averaging window length."""
        window = np.ones(int(window_size))/float(window_size)
        smoothederror = np.convolve(self.errorhist[start:end], window, 'valid')
        smoothedactivity = np.convolve(self.actfrachist[start:end], window, 'valid')
        plt.plot(smoothederror, 'b', label = 'Error')
        plt.plot(smoothedactivity, 'g', label = 'Activity')
        
    def pairwise_dots(self):
        """Compute all the pairwise dot products of the dictionary elements.
        Plot the resulting matrix."""
        phi = self.phi
        corr = phi @ phi.T
        plt.imshow(corr, cmap='RdBu', interpolation='nearest')
        plt.colorbar()  
        
    def cf_bandwidth_plot(self):
        """Each dictionary element determines a point on a plot of that element's
        bandwidth vs its center frequency."""
        centers, bandwidths = self.get_cf_and_bandwidth()
        plt.plot(centers, bandwidths, 'b.')
        plt.xlabel('Center frequency (Hz)')
        plt.xscale('log')
        plt.ylabel('Bandwidth (Hz)')
        plt.yscale('log')
        return centers, bandwidths
        
    def get_cf_and_bandwidth(self):
        spectra = np.square(np.abs(np.fft.rfft(self.phi, axis=1)))
        freqs = np.fft.fftfreq(self.lfilter, d=1/self.sample_rate)[:spectra.shape[1]]
        centers = spectra @ freqs / spectra.sum(1)
        bandwidths = np.sqrt(spectra @ freqs**2 / spectra.sum(1) - centers**2)
        return centers, bandwidths
        
    def revcorr(self, nstims=10, delay=100, whiten=True):
        RFs = np.zeros((self.nunits,self.lfilter+delay))
        stimcov = np.zeros((self.lfilter+delay,self.lfilter+delay))
        spikecounts = np.zeros(self.nunits)
        for nn in range(nstims):
            signal = self.stims.rand_stim()
            lsignal = signal.shape[0]
            spikes, recon = self.infer(signal)
            for tt in range(self.lfilter,lsignal-delay):
                segment = signal[tt-self.lfilter:tt+delay]
                RFs += np.outer(spikes[:,tt], segment)
                stimcov += np.outer(segment, segment)
                spikecounts += spikes[:,tt]
        #RFs /= spikecounts[:,None]
        if whiten:
            ridgeparam = 0.01
            RFs = np.linalg.lstsq(stimcov + ridgeparam*np.eye(self.lfilter+delay), RFs.T).T
        RFs = RFs/(np.linalg.norm(RFs,axis=1)[:,None])
        self.stims.tiled_plot(RFs)
        return RFs, spikecounts
        
    def write_dict_sounds(self, filename):
        phi = np.concatenate([self.phi, np.zeros((self.nunits,8000))],axis=1)
        sequence = phi.flatten()
        self.stims.write_sound(filename, sequence)
        
    # Sorting
    ##########################################################################
    def fast_sort(self, measure="L0", plot=False, savestr=None):
        """Sorts filters by moving average usage of specified type, or by center frequency.
        Options for measure: L0, L1, f. L0 by default."""
        if measure=="f" or measure=="frequency":
            usages, _ = self.get_cf_and_bandwidth()
        elif measure=="L1":
            usages = self.L1acts
        else:
            usages = self.L0acts
        sorter = np.argsort(usages)
        self.sort(usages, sorter, plot, savestr)
        return usages[sorter]
        
    def sort(self, usages, sorter, plot, savestr):
        self.phi = self.phi[sorter]
        self.L0acts = self.L0acts[sorter]
        self.L1acts = self.L1acts[sorter]
        self.L2acts = self.L2acts[sorter]
        if plot:
            plt.figure()
            plt.plot(usages[sorter])
            plt.title('L0 Usage')
            plt.xlabel('Dictionary index')
            plt.ylabel('Fraction of stimuli')
            if savestr is not None:
                plt.savefig(savestr,format='png', bbox_inches='tight')

    # Saving and loading
    ##########################################################################
    def save(self, filename=None):
        if filename is None:
            filename = self.paramfile
        with open(filename,'wb') as ff:
            pickle.dump([self.phi, self.params, self.histories], ff)
        self.paramfile = filename
        
    def load(self, filename):
        with open(filename, 'rb') as ff:
            self.phi, self.params, self.histories = pickle.load(ff)
        self.paramfile = filename
    
    # Properties for convenient access to groups of parameters and to tf variable values
    ##########################################################################    
    @property
    def phi(self):
        return self.sess.run(self.filters)
    @phi.setter
    def phi(self, phi):
        """Also updates masks."""
        self.sess.run(self.filters.assign(phi))
        self.masks = self.get_masks()
    
    @property
    def learn_rate(self):
        return self.sess.run(self._learn_rate)        
    @learn_rate.setter
    def learn_rate(self, new_rate):
        self.sess.run(self._learn_rate.assign(new_rate))
        
    @property
    def params(self):
        return {'thresh' : self.thresh,
                'normed_thresh' : self.normed_thresh,
                'learn_rate' : self.learn_rate}
    @params.setter
    def params(self, params):
        for key, val in params.items():
            try:
                getattr(self,key)
            except AttributeError:
                print('Unexpected parameter passed:' + key)
            setattr(self, key, val)
        
    @property
    def histories(self):
        return (self.errorhist, self.actfrachist, self.L0acts, self.L1acts, self.L2acts)     
    @histories.setter
    def histories(self, histories):
        self.errorhist, self.actfrachist, self.L0acts, self.L1acts, self.L2acts = histories