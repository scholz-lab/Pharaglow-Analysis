import os
import uuid
import errno
import pickle

from collections.abc import Mapping
from datetime import datetime
from matplotlib.pylab import style


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
