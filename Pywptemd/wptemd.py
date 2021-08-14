import numpy as np
from numpy.core.fromnumeric import argmax
import pywt
import emd.sift as emd
import pandas as pd
from scipy.stats import entropy
from sklearn.decomposition import FastICA

class wptemd:
    
    def __init__(self, **kwargs):
        # input arguments for wavelet packet decomposition:
        # wavelet = 'dmey' | Discrete Meyer (FIR Approximation)
        # maxlevel = 7 | decomposition levels
        # imf_opts a dict with the stop method and its parametes
        # n_components = 19 ICA number of components view Fig1
        self.pywt_params = {'wavelet':'dmey', 'maxlevel':7}
        self.emd_params = {'imf_opts':{'stop_method':'rilling', 'rilling_thresh':(0.05, 0.5, 0.05)}}
        self.imfselect_params = {'w':0.5}
        # rilling_thresh = (sd1, sd2, tol) or (Theta_1, Theta_2, alpha) eq(2-3)
        self.ica_params = {'n_components':19}
        if kwargs:
            self.pywt_params = {key:kwargs.get(key, self.pywt_params[key]) for key in self.pywt_params.keys()}
            self.emd_params = {key:kwargs.get(key, self.emd_params[key]) for key in self.emd_params.keys()}
            self.imfselect_params = {key:kwargs.get(key, self.imfselect_params[key]) for key in self.imfselect_params.keys()}
            self.ica_params = {key:kwargs.get(key, self.ica_params[key]) for key in self.ica_params.keys()}
            

    def _wpt_filter(self, data: np.array = np.random.randn(19,3000)) -> np.array:
        # input - data channels x N EEG measurement
        data = data
        channels, length = data.shape
        
        # wavelet packet computation for all channels
        wp = dict()
        for c in range(channels):
            x = data[c,:]
            wp[f'c_{c}'] = pywt.WaveletPacket(data=x, mode='symmetric', **self.pywt_params)
        
        # Node's energy filtering criterion - step 1 getting a dict with keys = nodes, 
        # values = matrix channels x N
        nodes = [node.path for node in wp['c_0'].get_level(7, 'natural')]
        wp_nodes = dict()
        for node in nodes:
            N = wp[f'c_{c}'][node].data.size
            wp_node = np.zeros([channels, N])
            for c in range(channels):
                wp_node[c,:] = wp[f'c_{c}'][node].data
            wp_nodes[node] = wp_node

        # Node's energy filtering criterion - step 2 computing energy criterion for node removal
        for key, value in wp_nodes.items():
            wp_nodes[key] = np.std(np.sum(value**2, axis = 1)) # eq (4) in paper
        max_value = max(wp_nodes.values())  # maximum value
        node_2_remove = [k for k, v in wp_nodes.items() if v == max_value]

        # Node's energy filtering criterion - step 3 removing node and reconstructing data
        filtered_data = np.zeros(data.shape)
        for key in wp.keys(): # dicts are unordered so not a good idea to use enumerate
            c = int(key.split('_')[-1])
            for node in node_2_remove:
                del wp[key][node]
            filtered_data[c,:] = wp[key].reconstruct()
        return filtered_data

    def wptemd_filter(self, data: np.array = np.random.randn(19,3000), \
        data_rest: np.array = np.random.randn(19,3000,)*.25) -> np.array:
        # input two EEG measurements of 19xN, data and data_rest
        # data is the measurement to filter (artifact removal)
        # data_rest is a measurement in rest state eyes closed, no movement, etc...
        channels = data.shape[0]

        # filtering data with wpt
        data = self._wpt_filter(data)

        # decomposing with emd data
        emd_data = {}
        for c in range(channels):
            x = data[c,:]
            emd_data[f'c_{c}'] = emd.sift(x, **self.emd_params)

        # decomposing with emd data_rest
        emd_rest = {}
        for c in range(channels):
            x = data_rest[c,:]
            emd_rest[f'c_{c}'] = emd.sift(x, **self.emd_params)
        
        # filtering each channel with J criterion
        data_filtered = np.zeros(data.shape)
        for key in emd_data.keys():
            c = int(key.split('_')[-1])
            data_filtered[c, :] =  self._imf_filter(emd_data[key], emd_rest[key], **self.imfselect_params)

        return data_filtered
        
        
    # filtering imf with entropy and std criterion eq(5)
    def _imf_filter(self, imfs_data: np.array, imfs_rest: np.array, w= 0.5) -> np.array:
        # input two np.array of imfs Nx # of imfs
        # imf_data emd decomposition of a given channel of the contaminated signal
        # imf_rest emd decomposition of a given channel for the resting signal
        K_imf = min([imfs_data.shape[-1], imfs_rest.shape[-1]]) # amount of imfs
        imfs_data = imfs_data[:, :K_imf]
        imfs_rest = imfs_rest[:, :K_imf]

        # computing criterion J eq5
        sigma = np.std(imfs_data, axis = 0)
        sigma_rest = np.std(imfs_rest, axis = 0)
        H = np.zeros(K_imf)
        H_rest = np.zeros(K_imf)

        for k in range(K_imf):
            H[k] = self._entropy_series(imfs_data[:,k])
            H_rest[k] = self._entropy_series(imfs_rest[:,k])

        J = w*(H/H_rest) + (1-w)*(sigma/sigma_rest)

        # imf removal and signal reconstruction
        imf_remove = np.argmax(J)
        imfs_data[:, imf_remove] = np.zeros(imfs_data.shape[0])
        return np.sum(imfs_data, axis= 1)

    @staticmethod
    def _entropy_series(data: np.array) -> float: # note in the paper they compute entropy over the IMF ???
        # input data np.array
        # output Shannon's entropy
        pd_series = pd.Series(data)
        counts = pd_series.value_counts()
        return entropy(counts) # entropy counts is entropy of a distribution for given probability values

    def wptica_filter(self, data: np.array = np.random.randn(19,3000)) -> np.array:
        # input data EEG measurements of 19xN with artifacts to be removed
        # Note that FastICA assumes Nx sources array so we have to transpose data before apply ICA
        channels = data.shape[0]

        # filtering data with wpt
        data = self._wpt_filter(data)

        # decomposing with ICA data
        ica = FastICA(random_state=0, **self.ica_params)
        ica_data = ica.fit_transform(data.T)

        # criterion to set to zero highest std component and filter signal
        sigma_c = np.std(ica_data, axis= 0)
        max_c = np.argmax(sigma_c)
        ica_data[:,max_c] = np.zeros(ica_data.shape[0])
        data_filtered = ica.inverse_transform(ica_data, copy=True)
        
        return data_filtered.T
if __name__ == '__main__':
    fs = 500
    data = np.random.randn(19,6*fs)
    data_rest = np.random.randn(19,6*fs)*.25

    parameters = {'wavelet':'dmey', 'maxlevel':7, # wavelet parameters
    'imf_opts':{'stop_method':'rilling', 'rilling_thresh':(0.05, 0.5, 0.05)}, # emd parameters
    'w':0.5, # emd selection criterion J parameter
    'n_components':19} # ICA parameter
    filter = wptemd(**parameters)

    wpt_filtered = filter._wpt_filter(data)
    print(wpt_filtered.shape)

    data_filtered_wptemd = filter.wptemd_filter(data, data_rest)
    print(data_filtered_wptemd.shape)

    data_filtered_wtpica = filter.wptica_filter(data)
    print(data_filtered_wtpica.shape)