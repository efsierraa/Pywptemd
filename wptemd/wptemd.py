import numpy as np
import pywt
import emd
from sklearn.decomposition import FastICA

class wptemd:
    
    def __init__(self, data: np.array = np.random.rand(19,3000)):
        self.data = data
        self.channels, self.length = data.shape

    def wpt_filtering(self,**kwargs):
        # input arguments for wavelet packet decomposition:
        # wavelet = 'dmey' | Discrete Meyer (FIR Approximation)
        # maxlevel = 7 | decomposition levels
        if kwargs:
            pywt_params = {key:value for key, value in kwargs.items()}
        else:
            pywt_params = {'wavelet':'dmey', 'maxlevel':7}
        
        # wavelet packet computation for all channels
        wp = dict()
        for c in range(self.channels):
            x = self.data[c,:]
            wp[f'c_{c}'] = pywt.WaveletPacket(data=x, mode='symmetric', **pywt_params)
        
        # Node's energy filtering criterion
        nodes = [node.path for node in wp['c_0'].get_level(7, 'natural')]
        print(len(nodes))
        wp_nodes = dict()
        for node in nodes:
            N = wp[f'c_{c}'][node].data.size
            wp_node = np.zeros([self.channels, N])
            for c in range(self.channels):
                wp_node[c,:] = wp[f'c_{c}'][node].data
            wp_nodes[node] = wp_node
        return wp_nodes

if __name__ == '__main__':
    fs = 500
    data = np.random.rand(19,6*fs)
    wptemd = wptemd(data)
    wp = wptemd.wpt_filtering()
    print(wp['a'*7].shape)