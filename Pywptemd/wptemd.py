import numpy as np
import pywt
import emd
from sklearn.decomposition import FastICA

class wptemd:
    
    def __init__(self, **kwargs):
        # input arguments for wavelet packet decomposition:
        # wavelet = 'dmey' | Discrete Meyer (FIR Approximation)
        # maxlevel = 7 | decomposition levels
        if kwargs:
            self.pywt_params = {key:value for key, value in kwargs.items()}
        else:
            self.pywt_params = {'wavelet':'dmey', 'maxlevel':7}

    def wpt_filter(self, data: np.array = np.random.randn(19,3000)) -> np.array:
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
        for c, key in enumerate(wp.keys()):
            for node in node_2_remove:
                del wp[key][node]
            filtered_data[c,:] = wp[key].reconstruct()
        return filtered_data

    def wptemd_filter(self, data: np.array = np.random.randn(19,3000), data_rest: np.array = np.random.randn(19,3000,)*.25) -> np.array:
        pass

if __name__ == '__main__':
    fs = 500
    data = np.random.randn(19,6*fs)
    wptemd_filter = wptemd()
    wpt_filtered = wptemd_filter.wpt_filter(data)
    print(wpt_filtered.shape)