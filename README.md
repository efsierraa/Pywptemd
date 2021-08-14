# WPTEMD
A class used to remove artifacts from EEG signals, the default EEG signal has a duration of 6 seconds and 19 channels.

## How to use it:
### Inputs:
EEG data with artifacts to be filtered, and EEG data in resting state, dimensions N x channels where N is the length of an EEG record
### Methods
_wpt_filter: filters data selecting the most corrupt node in a wpt removes it and makes the reconstruction of the signal, a wpt is computed per channel

wptemd_filter: applies _wpt_filter, and after decomposing each filtered channel into imfs using emd, to filter again the signal removing the most corrupt imf computing J a criterion based on entropy and std normalized with respect to the resting data

wptica_filter: like in wptemd_filter applies _wpt_filter to filter the data, afterward a second ICA based filter is applied where is removed the most contaminated ICA component, and reconstructed the sources
### Output
Data_filtered a filtered version of the original EEG data

### Example
```python
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
```


### References

```
[1] Bono, V., Das, S., Jamal, W., & Maharatna, K. (2016). Hybrid wavelet and EMD/ICA approach for artifact suppression in pervasive EEG. Journal of neuroscience methods, 267, 89-107.
```