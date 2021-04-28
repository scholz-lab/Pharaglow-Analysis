import os
import uuid
import errno
import pickle

import numpy as np


from collections.abc import Mapping
from datetime import datetime
from matplotlib.pylab import style
from scipy.signal import find_peaks


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


def preprocess(p, w_bg, w_sm, **kwargs):
    """preprocess a trace with rolling window brackground subtraction."""
    bg = p.rolling(w_bg, min_periods=1, center=True, win_type='hamming').mean()
    return (p - bg).rolling(w_sm, min_periods=1, center=True, win_type='parzen').mean(), bg


def find_pumps(p, heights = np.arange(0.01, 5, 0.1), min_distance = 5, sensitivity = 0.99, **kwargs):
    """peak detection in a background subtracted trace assuming real 
        peaks have to be at least min_distance samples apart."""
    tmp = []
    all_peaks = []
    # find peaks at different heights
    for h in heights:
        peaks = find_peaks(p, height = h,threshold = 0.0)[0]
        tmp.append([len(peaks), np.mean(np.diff(peaks)>=min_distance)])
        all_peaks.append(peaks)
    tmp = np.array(tmp)
    # set the valid peaks score to zero if no peaks are present
    tmp[:,1][~np.isfinite(tmp[:,1])]= 0
    # calculate random distribution of peaks in a series of length l (actually we know the intervals will be exponential)
    null = []
    l = len(p)
    for npeaks in tmp[:,0]:
        locs = np.random.randint(0,l,(500, int(npeaks)))
        # calculate the random error rate - and its stdev
        null.append([np.mean(np.diff(np.sort(locs), axis =1)>=min_distance), np.std(np.mean(np.diff(np.sort(locs), axis =1)>=5, axis =1))])
    null = np.array(null)
    # now find the best peak level - larger than random, with high accuracy
    # subtract random level plus 1 std:
    metric_random = tmp[:,1] - (null[:,0]+null[:,1])
    # check where this is still positive and where the valid intervals are 1 or some large value
    valid = np.where((metric_random>0)*(tmp[:,1]>=sensitivity))[0]
    if len(valid)>0:
        peaks = all_peaks[valid[np.argmax(tmp[:,0][valid])]]
    else:
        return [], tmp, null
    return peaks, tmp, null


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
