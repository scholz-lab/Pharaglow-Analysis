import os
import uuid
import errno
import pickle

import numpy as np
import pandas as pd

from collections.abc import Mapping
from datetime import datetime
from matplotlib.pylab import style
from scipy.signal import find_peaks, peak_prominences
from pyampd.ampd import find_peaks_adaptive


def hampel(vals_orig, k=7, t0=3):
    '''
    vals: pandas series of values from which to remove outliers
    k: size of window (including the sample; 7 is equal to 3 on either side of value)
    t0: how many sigma away to call it an outlier
    '''
    #Make copy so original not edited
    vals = vals_orig.copy()
    
    #Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(window=k, center=True, min_periods = 1).median()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True, min_periods=1).apply(MAD)
    threshold = t0 * L * rolling_MAD
    difference = np.abs(vals - rolling_median)
    outlier_idx = difference > threshold
    vals[outlier_idx] = rolling_median[outlier_idx] 
    return(vals)


def preprocess(p, w_bg, w_sm, win_type_bg = 'hamming', win_type_sm = 'boxcar', **kwargs):
    """preprocess a trace with rolling window brackground subtraction."""
    bg = p.rolling(w_bg, min_periods=1, center=True, win_type=win_type_bg).mean()
    return (p - bg).rolling(w_sm, min_periods=1, center=True, win_type=win_type_sm).mean(), bg


def illegal_intervals(signal, peaks, min_dist):
    """calculate the fraction of illegal intervals given a prominence cutoff."""
    prom,_,_  = peak_prominences(signal, peaks)
    frac_ill = np.zeros(len(prom))
    for i, p in enumerate(np.sort(prom)):
        tmp_peaks = peaks[prom>p]
        frac_ill[i] = np.sum(np.diff(tmp_peaks)<min_dist)/len(peaks)
    return prom, frac_ill


def select_valid_peaks(peaks, prom, frac_illegal, sensitivity):
    idx = np.where(frac_illegal <= 1-sensitivity)[0]
    if len(idx) >0:
        min_prom = np.sort(prom)[idx[0]]
        return peaks[prom>min_prom]
    else:
        return []


def _pyampd(signal, adaptive_window, min_distance = None, min_prominence = None, wlen = None):
    peaks = find_peaks_adaptive(signal, window=adaptive_window)
    # remove violating peaks by height or distance
    if min_prominence is not None:
        prom,_,_  = peak_prominences(signal, peaks, wlen)
        peaks = peaks[prom>min_prominence]

    if min_distance is not None:
        rejects = []
        prom,_,_ = peak_prominences(signal, peaks, wlen)
        # calculate peaks with violating intervals
        locs = np.where(np.diff(peaks)<min_distance)[0]
        #print(f'{len(locs)} violating peaks')
        # get the peaks around the offending interval
        for loc in locs:
            local_start = np.max([0, loc-3])
            local_end = np.min([loc+4, len(peaks)])
            local_peaks = peaks[local_start:local_end]
            local_prom = prom[local_start:local_end]
            local_diff = np.where(np.diff(local_peaks)<min_distance)[0]
            #plt.plot(signal[local_peaks[0]: local_peaks[-1]])
            #plt.plot(local_peaks-peaks[0], signal[local_peaks], 'ro')
            while len(local_diff  > 0):
                # remove smallest peak
                min_peak = local_peaks[np.argmin(local_prom)]
                rejects.append(min_peak)
                local_peaks = local_peaks[local_peaks != min_peak]
                local_prom = np.delete(local_prom, local_prom.argmin())
                # check if there are still offending intervals
                local_diff = np.where(np.diff(local_peaks)<min_distance)[0]

        rejects = np.unique(rejects)
        peaks = peaks[~np.isin(peaks, rejects)]
        locs = np.where(np.diff(peaks)<min_distance)[0]
    return peaks
    

def detect_peaks(signal, adaptive_window, min_distance = 4, min_prominence = None, sensitivity=0.95, use_pyampd = True, **kwargs):
    #peaks = find_peaks(signal,scale=adaptive_window)#_adaptive(signal, window=adaptive_window)
    if use_pyampd:
        peaks = _pyampd(signal, adaptive_window, min_prominence = min_prominence, **kwargs)
    else:
        peaks,_ = find_peaks(signal, prominence = min_prominence, **kwargs)
    if min_distance is not None:
        prominence, frac_illegal = illegal_intervals(signal, peaks, min_distance)
        peaks = select_valid_peaks(peaks, prominence, frac_illegal, sensitivity)
    return peaks
    


class PickleDumpLoadMixin:
    """ Provides methods to save an object to file and to load it from file

    API:
        load : Load class object from file
        dump : Save class object to file
    """
    #---------#
    # Private #
    #---------#
    def _set_path(self, file_path):
        """ Setting the object path, used when the object was loaded from a pickle file """
        self._pkl_path = file_path

    def _get_path(self):
        """ Makes the object path available if the object was loaded from a pickle file """
        if hasattr(self, '_pkl_path'):
            return self._pkl_path
        else:
            return None

    #--------#
    # Public #
    #--------#
    @classmethod
    def load(cls, file_path):
        """ Load  object from file

        Args:
            file_path (str) : Path to the normalizer pickle file

        Returns:
            Instance of the given class from pickle file
        """
        with open(file_path, 'rb') as f_in:
            obj = pickle.load(f_in)
            assert isinstance(obj, cls), f'Object in {file_path} is not of type {cls}'
            obj._set_path(file_path)
            return obj

    def dump(self, file_path):
        """ Save object in its current state to file

        Args:
            file_path (str) : Path to the output pickle file

        Returns:
            -
        """
        with open(file_path, 'wb') as f_out:
            self._set_path(file_path)
            pickle.dump(self, f_out)


def lad_mplstyle():
    """ Plotting style for the evaluator """
    top_dir = os.path.dirname(os.path.abspath(__file__))
    return style.context(os.path.join(top_dir, 'lad.mplstyle'))

def check_for_none_str(s):
    """ Used to parse str Nones to proper None types - A shortcoming of TOML """
    if s == 'None':
        return None
    return s

def get_last_subdir(directory):
    """ Find the most recent subdir following our timestamp convention """
    dirs    = [f.name for f in os.scandir(directory) if f.is_dir()]
    subdirs = []
    for d in dirs:
        try:
            subdirs.append((datetime.strptime(d[:15], '%Y%m%d_%H%M%S'), d))
        except ValueError:
            pass
    subdirs = sorted(subdirs, key=lambda x: x[0])
    return os.path.join(directory, subdirs[-1][1])

def safe_make_dir(directory):
    """ Make dir while trying to avoid race conditions """
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def flatten_dict(d, prefix=''):
    """ Flatten the key structure of a dictionary """
    out = []
    for k in d.keys():
        if isinstance(d[k], dict):
            for item in flatten_dict(d[k], prefix=f'{prefix}{k}|'):
                out.append(item)
        else:
            out.append([f'{prefix}{k}', d[k]])
    return out

def dict_from_flat_key(key, value):
    """ Reconstruct a dictionary from flattened structure """
    d = {}
    keys = key.split('|', 1)
    if len(keys) == 1:
        d = {keys[0]: value}
        return d
    else:
        d[keys[0]] = dict_from_flat_key(keys[1], value)
        return d

def update_nested_dict(d, u):
    """ Update one nested dict with another """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_nested_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def get_uid():
    """ Create a unique ID """
    return str(uuid.uuid4())[:8]

def set_token(token_dir):
    """ Set a token, symbolizing a process has finished """
    safe_make_dir(token_dir)
    token = open(os.path.join(token_dir, 'done.token'), 'w')
    token.close()
    return
    
def check_token(token_dir):
    """ Check for a token, symbolizing the process has finished """
    return os.path.isfile(os.path.join(token_dir, 'done.token'))

def calc_metric(tmp, metric, axis=0, key=None):
    """Calls metric calculation on pandas Dataframe or Series"""
    kwargs = {}
    if axis is not None:
        kwargs = {'axis':axis}

    if metric == None:
        return tmp
    if metric == "sum":
        return tmp.sum(**kwargs)
    if metric == "mean":
        return tmp.mean(**kwargs)
    if metric == "std":
        return tmp.std(**kwargs)
    if metric == "N":
        return tmp.count(**kwargs)
    if metric == "sem":
        return tmp.std(**kwargs)/np.sqrt(tmp.count(**kwargs))
    if metric == "median":
        return tmp.median(**kwargs)
    if metric == "rate":
        return tmp.sum(**kwargs)/tmp.count(**kwargs)
    if metric == 'max':
        return tmp.max(**kwargs)
    if metric == 'min':
        return tmp.min(**kwargs)
    if metric == "collapse":
        return pd.DataFrame(tmp.values.ravel(), columns = [key])
    else:
        raise Exception("Metric not implemented, choose one of 'mean', 'median', 'std', 'sem' , 'sum', 'rate', 'median', 'max', 'min', 'N' or 'collapse'")

