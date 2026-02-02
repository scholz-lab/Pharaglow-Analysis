import os
import pickle
import copy
import warnings
import yaml
import json

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.stats import circmean, circstd
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path

from . import style
from . import tools
from .tools import PickleDumpLoadMixin


# Default units definition for PharaGlow data columns
UNITS = {
    'x': 'px',
    'y': 'px',
    'x_scaled': 'um',
    'y_scaled': 'um',
    'frame': '1',
    'time': 's',
    'time_align': 's',
    'time_aligned': 's',
    'pumps': 'a.f.u.',
    'pumps_clean': 'a.f.u.',
    'pump_events': '1',
    'rate': '1/s',
    'count_rate': '1/s',
    'count_rate_pump_events': '1/s',
    'velocity': 'um/s',
    'velocity_smooth': 'um/s',
    'nose_speed': 'um/s',
    'cms_speed': 'um/s',
    'reversals': '1',
    'reversals_nose': '1',
    'inside': '1',
    'Imean': 'a.f.u.',
    'Imax': 'a.f.u.',
    'Istd': 'a.f.u.',
    'skew': '1',
    'area': 'px^2',
    'Area2': 'px^2',
    'size': 'mm',
    'Centerline': '1',
    'centerline_scaled': 'um',
    'Straightened': '1',
    'temperature': 'C',
    'humidity': '%',
    'age': 'h',
    '@acclimation': 'min',
    'particle': '1',
    'image_index': '1',
    'im_idx': '1',
    'has_image': '1',
    'index': '1',
    'space_units': 'um',
    'time_units': 's',
}



def fast_window(x, w, min_periods_one=True, anchor='back'):
    """
    Returns a sliding window view similar as pandas rolling, but based on numpy sliding_window_view
    Args:
        x (numpy.ndarray): input data, must be 1 dimensional
        w (int): width of sliding window 
        min_periods_one (bool): minimum number of observations in window required to have a value is one
        anchor (str): Where to anchor the sliding window
    Returns:
        v (np.ndarray): arry containing sliding window in rows
    """
    v = sliding_window_view(x, w)
    if not min_periods_one:
        return v

    min_s = np.ones((w,w))*x[:w]
    min_s[np.triu_indices(min_s.shape[0],1)] = np.nan
    min_e = np.ones((w,w))*x[-w:]
    min_e[np.tril_indices(min_e.shape[0],-1)] = np.nan
    
    if anchor not in ['back','front','center']:
        raise ValueError(f"anchor must be one of ['back','front','center'], but is '{anchor}'")
    if anchor=='back':
        min_s = min_s[:-1]
        v=np.vstack([min_s,v])
    if anchor=='front':
        min_e = min_e[1:]
        v=np.vstack([v,min_e])
    elif anchor=='center':
        min_s = min_s[w//2:-1]
        min_e = min_e[1:(w//2)+1]
        v=np.vstack([min_s,v,min_e])
    return v

def _lineplot(x ,y, yerr, ax, **kwargs):
    plot = []
    if isinstance(ax, list):
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        for wi in range(x.shape[1]):
            xi = x.iloc[:,wi]
            yi = y.iloc[:,wi]
            if wi>len(ax):
                warnings.warn('Too few subplots detected. Multiple samples will be plotted in a subplot.')
            plot.append(ax[(wi)%len(ax)].plot(xi.values, yi.values, **kwargs))
            if yerr is not None:
                yerr_i = yerr.iloc[:,wi]
                alpha = kwargs.pop('alpha', 0.5)
                ax[(wi)%len(ax)].fill_between(xi.values, yi.values-yerr_i.values, yi.values+yerr_i.values, alpha = alpha, **kwargs)
    else:
        plot = ax.plot(x.values, y.values, **kwargs)
        if yerr is not None:
            alpha = kwargs.pop('alpha', 0.5)
            ax.fill_between(x.values, y.values-yerr.values, y.values+yerr.values, alpha = alpha, lw=0,  **kwargs)
    return plot


def _hist(y, ax, **kwargs):
    plot = []
    y = pd.DataFrame(y)
    if isinstance(ax, list):
        for wi in range(y.shape[1]):
            yi = y.iloc[:,wi]
            if wi>len(ax):
                warnings.warn('Too few subplots detected. Multiple samples will be plotted in a subplot.')
                # the histogram of the data
            num_bins = kwargs.pop('nbins', int(yi.count()**0.5))
            density = kwargs.pop('density', True)
            n, bins, patches = ax[(wi)%len(ax)].hist(yi, num_bins, density=density, **kwargs)
            kwargs['nbins'] = num_bins
            kwargs['density'] = density
            #plot.append(ax[(wi+1)%len(ax)].plot(bins, yi, **kwargs))
    else:
        if len(y.columns)>1:
            # if more than one dataset
            for wi in range(y.shape[1]):
                yi = y.iloc[:,wi]
                # the histogram of the data
                num_bins = kwargs.pop('nbins', int(yi.count()**0.5))
                density = kwargs.pop('density', True)
                n, bins, patches = ax.hist(yi, num_bins, density=density, **kwargs)
                kwargs['nbins'] = num_bins
                kwargs['density'] = density
        else:
            # the histogram of the data
            num_bins = kwargs.pop('nbins', int(y.count()**0.5))
            density = kwargs.pop('density', True)
            plot = ax.hist(y, num_bins, density=density, **kwargs)
            #plot = ax.plot(bins, y, **kwargs)
    return plot


def _scatter(x, y, xerr, yerr, ax, density = False, **kwargs):
    plot = []
    linestyle = kwargs.pop('linestyle', "none")
    marker = kwargs.pop('marker', 'o')
    if isinstance(ax, list):
        # make sure we have two dataframes, not series
        x = pd.DataFrame(x)
        y = pd.DataFrame(y)
        
        for wi in range(x.shape[1]):
            xi = x.iloc[:,wi]
            yi = y.iloc[:,wi]
            
            if wi>len(ax):
                warnings.warn('Too few subplots detected. Multiple samples will be plotted in a subplot.')
            if density:
                filt = np.isfinite(xi)*np.isfinite(yi)
                plot.append(style.KDE_plot(ax[(wi+1)%len(ax)], x[filt],yi[filt], **kwargs))
            else:
                if yerr is not None and xerr is not None:
                    plot.append(ax[(wi)%len(ax)].errorbar(xi.values, yi.values, yerr.iloc[:,wi].values, xerr.iloc[:,wi].values, linestyle = linestyle, marker = marker, **kwargs))
                elif yerr is not None:
                    plot.append(ax[(wi)%len(ax)].errorbar(xi.values, yi.values, yerr.iloc[:,wi].values, xerr.iloc[:,wi].values, linestyle = linestyle, marker = marker,  **kwargs))
                else:
                    plot.append(ax[(wi)%len(ax)].scatter(xi.values, yi.values, **kwargs))
    else:
        if density:
            filt = np.isfinite(x)*np.isfinite(y)
            style.KDE_plot(ax, x[filt],y[filt], **kwargs)
        else:
            if yerr is not None:
                plot = ax.errorbar(x.values, y.values, yerr.values, xerr.values,  linestyle = linestyle, marker = marker, **kwargs)
            else:
                plot = ax.scatter(x.values, y.values, **kwargs)
    return plot


def _heatmap(x, y, ax, **kwargs):
    plot = []
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    if isinstance(ax, list):
        for wi in range(y.shape[1]):
            yi = y.iloc[:,wi]
            yi = pd.DataFrame(yi)
            xi = x.iloc[:,wi]
            if wi>len(ax):
                warnings.warn('Too few subplots detected. Multiple samples will be plotted in a subplot.')
                # a heatmap of the data
            im = ax[(wi)%len(ax)].imshow(yi.values.T, **kwargs)
            plot.append(im)
    else:
        # a heatmap of the data
        im = ax.imshow(y.values.T, **kwargs)
        plot.append(im)
    return plot



class Worm(PickleDumpLoadMixin):
    """class to contain data from a single pharaglow result."""
    def __init__(self, filename, columns, fps, scale, units, particle_index = None, load = True, **kwargs):
        """initialize object and load a pharaglow results file."""
        self.fps = fps
        self.scale = scale
        self.flag = False
       
        # keep some metadata
        self.experiment = os.path.basename(filename)
        if particle_index is not None:
            self.particle_index = particle_index 
        else:
            self.particle_index = int(os.path.splitext(self.experiment)[0].split('_')[-1])
        # units
        self.columns = columns
        self.units = units
        # load data
        if load:
            print('Reading', filename)
            self._load(filename, columns, fps, scale, **kwargs)
            
            
    def _load_CA(self, filename, columns, fps, scale, **kwargs): # ER
        """load data from my calcium imaging of the pharynx."""
        with open(filename) as f:
            tmp = json.load(f)
        traj = pd.DataFrame(tmp)
        # drop all columns except the ones we want - but keep the minimal values
        traj = traj.filter(columns)
        # extract the centerlines and other non-scalar values into an array instead
        if 'Centerline' in columns:
            self.centerline = np.array([np.array(cl) for cl in traj['Centerline']])
        if 'Straightened' in columns:
            self.images = np.array([np.array(im) for im in traj['Straightened']])
        traj = traj.drop(['Centerline', 'Straightened'], errors = 'ignore')
         
        
        
    def _load(self, filename, columns, fps, scale, **kwargs):
        """load data."""
        traj = pd.read_json(filename, orient='split', **kwargs)
        # drop all columns except the ones we want - but keep the minimal values
        traj = traj.filter(columns)
        # extract the centerlines and other non-scalar values into an array instead
        if 'Centerline' in columns:
            self.centerline = np.array([np.array(cl) for cl in traj['Centerline']])
        if 'Straightened' in columns:
            self.images = np.array([np.array(im) for im in traj['Straightened']])
        traj = traj.drop(['Centerline', 'Straightened'], errors = 'ignore')
       
        self.data = traj
        self.data = self.data.reset_index()


    def __repr__(self):
        return f"Worm \n with underlying data: {self.data.describe()}"


    def __len__(self):
        return len(self.data)
   
        
    ######################################
    #
    #   get/set attributes
    #
    #######################################
    def create_ID(self):
        """create a unique ID matching raw data"""
        self.id = f"{self.experiment}_{self.particle_index}"
        
        
    def get_metric(self, key, metric, filterfunction = None):
        """return metrics of a data column given by key.
            filterfunction: a callable that returns a boolean for each entry in the series data[key] 
        """
        assert key in self.data.columns, f'The key {key} does not exist in the data.'
        tmp = self.data.set_index('frame')[key]
        if filterfunction is not None:
            filtercondition = filterfunction(tmp)
            tmp = tmp.loc[filtercondition]

        if metric == "sum":
            return tmp.sum()
        if metric == "mean":
            return tmp.mean()
        if metric == "std":
            return tmp.std()
        if metric == "N":
            return tmp.count()
        if metric == "sem":
            return tmp.std()/np.sqrt(tmp.count())
        if metric == "median":
            return tmp.median()
        if metric == "rate":
            return tmp.sum()/tmp.count()*self.fps
        if metric == 'max':
            return tmp.max()
        if metric == 'min':
            return tmp.min()
        else:
            raise Exception("Metric not implemented, choose one of 'mean','median', 'std', 'sem' , 'sum', 'rate','median', 'max', 'min' or 'N'")
    
    
    def get_aligned_metric(self, key, metric, filterfunction = None):
        """get averages across timepoints for a single worm. eg. average across multiple stimuli. 
        Requires multi_align(self) to be run.

        filterfunction: a callable that returns a boolean for each entry in the series aligned_data[key] 
        """
        assert len(self.aligned_data)>0, 'Please run Worm.align() or Worm.multi_align() first!'
        assert key in self.aligned_data[0].columns, f'The key {key} does not exist in the data.'
        tmp = self.get_data_aligned(key)
        if filterfunction is not None:
            filtercondition = filterfunction(tmp)
            tmp = tmp.loc[filtercondition]
        if metric == "sum":
            return tmp.sum(axis = 1)
        if metric == "mean":
            return tmp.mean(axis = 1)
        if metric == "std":
            return tmp.std(axis = 1)
        if metric == "N":
            return tmp.count(axis =1)
        if metric == "sem":
            return tmp.std(axis = 1)/np.sqrt(tmp.count(axis=1))
        if metric == 'median':
            return tmp.median(axis = 1)
        if metric == 'max':
            return tmp.max(axis = 1)
        if metric == 'min':
            return tmp.min(axis = 1)
        else:
            raise Exception("Metric not implemented, choose one of 'mean', 'median', 'std', 'sem', 'sum,','median', 'max', 'min' or 'N'")


    def get_data(self, key = None, aligned = False, index_column = 'frame'):
        """return a column of data with name 'key' or the whole pandas dataframe.
        key: a column of self.data
        aligned: Toggle using aligned or full data
        index_column: The returned dataframe or series will have that column as index. Important for concatenation.
        """
        if aligned:
            self.get_data_aligned(key)
        if key == None:
            return self.data.set_index(index_column)
        elif key == 'frame':
            warnings.warn(f'You requested {key}. The index of this series will be meaningless.')
            tmp = self.data['frame']
            tmp.index = self.data['frame'].values
            return tmp
        else:
            assert key in self.data.columns, f'The key {key} does not exist in the data.'
            return self.data.set_index(index_column)[key]


    def get_data_aligned(self, key = None):
        """return a column of aligned data or the whole aligned pandas dataframe at a specific timepoints."""
        assert len(self.aligned_data)>0, 'Please run Worm.align() or Worm.multi_align() first!'
        if key == None:
            return self.aligned_data
        else:
            assert key in self.aligned_data[0].columns, f'The key {key} does not exist in the data.'
            return pd.concat([data.loc[:,key] for data in self.aligned_data], axis = 1)


    def get_events(self, events = 'pump_events', unit = 'index', aligned = False):
        """return peak locations for this worm i.e., where a binary column is 1 or True.
        unit: column of data at which to evaluate e.g. 'time'
        """
        if aligned:
            assert len(self.aligned_data)>0, 'Please run Worm.align() or Worm.multi_align() first!'
            for subset in self.aligned_data:
                tmp = []
                if unit =='index' or unit == None:
                    tmp.append(subset.index[subset[events]==True].values)
                else:
                    tmp.append(subset[unit][subset[events]==True].values)
            return tmp
        else:
            if unit =='index' or unit == None:
                return self.data.index[self.data[events]==True].values
            else:
                return self.data[unit][self.data[events]==True].values


    def add_column(self, key, values, overwrite = True):
        """add a data column to the underlying datset. Will overwrite existing column names if overwrite.
            key: name of new (or existing) column
            values: list with the same length or series with matching index, or dict with ma
            overwrite: replaces column if it exists.
        """

        if key in self.data.columns and not overwrite:
            warnings.warn(f'Column {key} exists. If you want to overwrite the existing data, use overwrite = True.')
        else:
            if len(values) == len(self.data.index):
                self.data[key] = values
            else:
                warnings.warn(f'Length of values {len(values)} does not match size of the data {len(self.data.index)}. Column was not updated.')
    
    ######################################
    #
    #   calculate additional metrics
    #
    #######################################
    def preprocess_signal(self, key, w_outlier, w_bg, w_smooth, **kwargs):
        "Use outlier removal, background subtraction and filtering to clean a signal."
        # remove outliers
        sigma = kwargs.pop('sigma', 3)
        # make a copy of the signal
        self.data.loc[:,f'{key}_clean'] = self.data[key]
        if w_outlier is not None:
            self.data[f'{key}_clean'] = tools.hampel(self.data[f'{key}_clean'], w_outlier, sigma)
        self.data[f'{key}_clean'],_ = tools.preprocess(self.data[f'{key}_clean'], w_bg, w_smooth)
        self.units[f'{key}_clean'] = self.units[key]

        
    def calculate_property(self, name, **kwargs):
        """calculate additional properties on the whole dataset, calling functions for each worm."""

        funcs = {"reversals": self.calculate_reversals,
                 "count_rate": self.calculate_count_rate,
                 "smoothed": self.calculate_smoothed,
                 "pumps": self.calculate_pumps,
                 "nose_speed": self.calculate_nose_speed,
                 "reversals_nose": self.calculate_reversals_nose,
                 "velocity":self.calculate_velocity,
                 "time":self.calculate_time,
                 "preprocess_signal": self.preprocess_signal,
                 "locations": self.calculate_locations,
                }
        if name == 'help':
            print(funcs.keys())
            return
        # run function
        funcs[name](**kwargs)
        # update columns
        self.columns = self.data.columns


    def calculate_smoothed(self, key, window, aligned = False, **kwargs):
        """use rolling apply to smooth a series with the given parameters which is added as a column to the existing data.
        window: size of the rolling window.
        key: a column of self.data or self.aligned_data
        kwargs: passed onto pd.rolling
        """
        kwargs['win_type'] = kwargs.pop('win_type', 'boxcar')
        kwargs['center'] =  kwargs.pop('center', True)
        kwargs['min_periods'] =  kwargs.pop('min_periods', 1)
        if aligned:
            assert len(self.aligned_data)>0, 'Please run Worm.align() or Worm.multi_align() first!'
            for dset in self.aligned_data:
                dset[f'{key}_smoothed'] = dset[key].rolling(window, **kwargs).mean()
        else:
            self.data[f'{key}_smoothed'] = self.data[key].rolling(window, **kwargs).mean()
            self.units[f'{key}_smoothed'] = self.units[key]

            
            
    def calculate_time(self):
        """calculate time from frame index."""
        # real time
        self.data.loc[:,'time'] = self.data['frame']/self.fps
        self.units['time'] = self.units['time_units']
    
    
    def calculate_locations(self):
        """calculate correctly scaled x,y coordinates."""
        # real time
        self.data.loc[:,'x_scaled'] = self.data['x']*self.scale
        self.data.loc[:,'y_scaled'] = self.data['y']*self.scale
        self.units['x_scaled'] = self.units['space_units']
        self.units['y_scaled'] = self.units['space_units']
        try:
            self.centerline_scaled = self.centerline*self.scale
            self.units['centerline_scaled'] = self.units['space_units']
        except AttributeError:
            pass
            
        
    def calculate_velocity(self, units=None, dt = 1, columns = ['x', 'y']):
        """calculate velocity from the coordinates."""
        try:
            cms = np.stack([self.data[columns[0]], self.data[columns[1]]]).T
            v_cms = cms[dt:]-cms[:-dt]
            t = np.array(self.data.frame)
            deltat = t[dt:]-t[:-dt]
            velocity = np.sqrt(np.sum((v_cms)**2, axis = 1))/deltat*self.scale*self.fps
            velocity = np.append(velocity, [np.nan]*dt)
            #velocity= np.sqrt((self.data['x'].diff()**2+self.data['y'].diff()**2))/self.data['frame'].diff()*self.scale*self.fps
            self.data['velocity'] = velocity
            
            if units is None:
                units = f"{self.units['space_units']}/{self.units['time_units']}"
            self.units['velocity'] = units

        except KeyError:
            print('Velocity calculation failed. Continuing.')
        
        
    def calculate_pumps(self, min_distance,  sensitivity, adaptive_window, min_prominence = 0, key = 'pump_clean', use_pyampd = True):
        """using a pump trace, get additional pumping metrics."""
        signal = self.data[key]
        peaks = tools.detect_peaks(signal, adaptive_window, min_distance, min_prominence, sensitivity, use_pyampd)
        if len(peaks)>1:
            # add interpolated pumping rate to dataframe
            self.data.loc[:,'rate'] = np.interp(np.arange(len(self.data)), peaks[:-1], self.fps/np.diff(peaks))
            # # get a binary trace where pumps are 1 and non-pumps are 0
            self.data.loc[:,'pump_events'] = 0
            self.data.loc[peaks,['pump_events']] = 1
        else:
            self.data.loc[:,'rate'] = 0
            self.data.loc[:,'pump_events'] = 0
        self.units['rate'] = f"1/{self.units['time']}"
        self.units['pump_events'] = "1"

        
    def calculate_count_rate(self, window, key='pump_events', **kwargs):
        """Add a column 'count_rate' to self.data. Calculate a rate based on the number of binary events in a window. 
        window is in frame. Result will be in Hz."""
        kwargs['center'] =  kwargs.pop('center', True)
        kwargs['min_periods'] =  kwargs.pop('min_periods', 1)
        self.data[f'count_rate_{key}'] = self.data[key].rolling(window, **kwargs).sum()/window*self.fps
        self.units[f'count_rate_{key}'] = f"{self.units[key]}/{self.units['time']}"


    def calculate_reversals(self, animal_size, angle_threshold, scale = None):
        """Adaptation of the Hardaker's method to detect reversal event. 
        A single worm's centroid trajectory is re-sampled with a distance interval equivalent to 1/10 
        of the worm's length (100um) and then reversals are calculated from turning angles.
        Inputs:
            animal_size: animal size in um.
            angle_threshold (degree): what defines a turn 
        Output: None, but adds a column 'reversals' to  self.data.
        """
        # check if the x,y coordintaes need to be rescaled to um later
        if scale is None:
            scale = self.scale
        # resampling
        # Calculate the distance cover by the centroid of the worm between two frames um
        cummul_distance = np.cumsum(self.data['velocity'])*self.data['time'].diff()
        # create an array of distances that are animal_size appart for sampling,
        # find the maximum number of worm lengths we have travelled
        maxlen = cummul_distance.max()/animal_size
        sample_distance = np.arange(animal_size, maxlen*animal_size, animal_size)
        # for each sample distance select the closest value from cummulative distance
        sample_indices = pd.DataFrame(abs(cummul_distance.values - sample_distance[:, np.newaxis])).T
        sample_indices = sample_indices.idxmin(axis=0).values

        # create a downsampled trajectory from these indices
        traj_Resampled = self.data.loc[sample_indices, ['x', 'y']].diff()*scale
        # we ignore the index here for the shifted data
        traj_Resampled[['x1', 'y1']] = traj_Resampled.shift(1).fillna(0)

        # use the dot product to calculate the andle
        def angle(row):
            old_err = np.seterr(divide='ignore', invalid='ignore')
            v1 = [row.x, row.y]
            v2 = [row.x1, row.y1]
            deg = np.degrees(np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)))
            np.seterr(**old_err)
            return deg
        traj_Resampled['angle'] = 0
        traj_Resampled['angle']= traj_Resampled.apply(lambda row: angle(row), axis=1)
        
        rev = traj_Resampled.index[traj_Resampled.angle>=angle_threshold]
        #self.data.loc[:,'angle'] = traj_Resampled['angle']
        self.data.loc[:,'reversals'] = 0
        self.data.loc[rev,'reversals'] = 1
        # units
        self.units['reversals'] = '1'
     

    def calculate_reversals_nose(self, dt = 1, angle_threshold = 150, w_smooth = 30, min_duration = 30):
        """using the motion of the nosetip relative to the center of mass motion to determine reversals."""
        
        try:
            cl = self.centerline
        except AttributeError:
            warnings.warn('data does not contain centerlines.')
            return
        cms = np.stack([self.data.x, self.data.y]).T
        # trajectories - CMS - coarse-grain
        # check the number of points of each centerline
        nPts = len(cl[0])
        # note, the centerline is ordered y,x
        # subtract the mean since the cl is at (50,50) or similar
        yc, xc = cl.T - np.mean(cl.T, axis = 1)[:,np.newaxis]
        cl_new = np.stack([xc, yc]).T + np.repeat(cms[:,np.newaxis,:], nPts, axis = 1)

        old_err = np.seterr(divide='ignore', invalid='ignore') # ignore invalid and divide errors during angle calcuation
        # extract direction of worm/pharynx over space
        nose_vec = cl_new[dt:,0]-cl_new[:-dt,0] # movement of the nose
        nose_vlen = np.linalg.norm(nose_vec, axis=1)[:,np.newaxis]
        nose_unit = np.divide(nose_vec,nose_vlen)
        nose_unit = np.nan_to_num(nose_unit)
        # extract direction of cms over time using dt
        heading_vec = cl_new[:,0]-cl_new[:,nPts//2] # tangent of the worm = 'heading'
        heading_vlen = np.linalg.norm(heading_vec, axis=1)[:,np.newaxis]
        heading_unit = np.divide(heading_vec,heading_vlen)
        heading_unit = np.nan_to_num(heading_unit)
        np.seterr(**old_err)
        
        # angle relative to cms motion
        crop = min(len(nose_unit), len(heading_unit))
        dotProduct = nose_unit[:crop,0]*heading_unit[:crop,0] +nose_unit[:crop,1]*heading_unit[:crop,1]
        angle = np.rad2deg(np.arccos(dotProduct))

        # smooth angle with fast_window to increase speed
        angle = circmean(fast_window(angle, w_smooth, anchor='center'), 180, 0, axis=1, nan_policy='omit')
        angle[np.isnan(angle)] = 0
        angle = np.append(angle, [np.nan]*dt)
        self.add_column('angle_nose', angle, overwrite = True)
        # determine when angle is over threshold
        rev = angle > angle_threshold
        # filter short reversals
        rev = pd.Series(rev).rolling(2*min_duration, center=True).median()
        rev = rev>0
        #rev = np.append(rev, [np.nan]*dt)
        self.add_column('reversals_nose', rev, overwrite = True)
        # add reversal events (1 where a reversal starts)
        reversal_start = np.diff(np.array(rev, dtype=int))==1
        reversal_start = np.append(reversal_start, [0])
        self.add_column('reversal_events_nose', reversal_start, overwrite = True)
        # add units
        self.units['angle_nose'] = 'degrees'
        self.units['reversal_events_nose'] = '1'
        self.units['reversals_nose'] = '1'
        

    def calculate_nose_speed(self, dt = 1):
        """Calculate the cms and nose velocities. """
        try:
            cl = self.centerline
        except AttributeError:
            warnings.warn('data does not contain centerlines.')
            return
        # trajectories - CMS - coarse-grain
        cms = np.stack([self.data.x, self.data.y]).T
        # note, the centerline is ordered y,x
        # subtract the mean since the cl is at (50,50)
        yc, xc = cl.T - np.mean(cl.T, axis = 1)[:,np.newaxis]
        cl_new = np.stack([xc, yc]).T + np.repeat(cms[:,np.newaxis,:], 100, axis = 1)
        nose = cl_new[:,0,:]
        #calculate directions - in the lab frame! and with dt
        v_nose = nose[dt:]-nose[:-dt]
        v_cms = cms[dt:]-cms[:-dt]
        t = np.array(self.data.time)
        deltat = t[dt:]-t[:-dt]
        v_nose_abs = np.sqrt(np.sum((v_nose)**2, axis = 1))/deltat*self.scale#*self.fps
        v_cms_abs = np.sqrt(np.sum((v_cms)**2, axis = 1))/deltat*self.scale#*self.fps
        # add back the missing item from difference
        v_nose_abs = np.append(v_nose_abs, [np.nan]*dt)
        v_cms_abs = np.append(v_cms_abs, [np.nan]*dt)
        # add to data
        self.add_column('nose_speed', v_nose_abs, overwrite = True)
        self.add_column('cms_speed', v_cms_abs, overwrite = True)
        # add units
        self.units['nose_speed'] = f"{self.units['space_units']}/{self.units['time_units']}"
        self.units['cms_speed'] = f"{self.units['space_units']}/{self.units['time_units']}"

        
    def align(self, timepoint,  tau_before, tau_after, key = None, column_align = 'frame'): # , **kwargs)
        """align to a timepoint.
         Inputs:
                timepoint: time to align to in frames
                tau_before: number of frames before timepoint
                tau_after: number of frames after timepoint
                key is a string or list of strings as column names.
        Output: creates a list of aligned dataframes centered around the timepoint 
        """
        if key is None:
            key = self.data.columns
        # create a list of desired elements and then chunk a piece of data around them
        tstart, tend = timepoint - tau_before, timepoint + tau_after
        frames = np.arange(tstart, tend + 1)
        tmp = self.data[self.data[column_align].isin(frames)].loc[:,key]
        # fill missing data
        tmp = tmp.set_index(column_align)
        tmp = tmp.reindex(pd.Index(frames))
        tmp.index = pd.Index(np.arange(-tau_before, tau_after + 1))
        tmp['time_aligned'] = tmp.index.values/self.fps
        
#         rescale_time = kwargs.pop(' rescale_time ', ('Time' in column_align)|('time' in column_align))
        
#         if rescale_time:
#             tmp['time_align'] = tmp.index.values/self.fps
#         else:
#             tmp['time_align'] = tmp.index.values
        return tmp
    

    def multi_align(self, timepoints, tau_before, tau_after, key = None, column_align = 'frame'):
        """align to multiple timepoints.
         Inputs:
                timepoints: list of timepoints to align to in frames
                tau_before: number of frames before timepoint
                tau_after: number of frames after timepoint
                key is a string or list of strings as column names.
        Output: creates a dictionary of aligned dataframes centered around the timepoint 
        """
        self.aligned_data = []
        self.timepoints = timepoints
        if key == None:
            key = self.data.columns
        for timepoint in self.timepoints:
            tmp = self.align(timepoint,  tau_before, tau_after, key, column_align)
            self.aligned_data.append(tmp)

    
class Experiment(PickleDumpLoadMixin):
    """Wrapper class which is a container for individual worms."""
    # class attributes
    def __init__(self, strain, condition, scale, fps, scale_units = None, fps_units = None, samples = None, color = None):
        self.strain = strain
        self.condition = condition
        self.scale = scale
        self.fps = fps
        # a place for detection parameters, ...
        self.metadata = {}
        if samples == None:
            self.samples = []
        else:
            self.samples = samples[:]
        self.color = color
        # units
        if scale_units is None:
            self.space_units = 'um'
        else:
            self.space_units = scale_units
        if fps_units is None:
            self.time_units = 's'
        else:
            self.time_units =  fps_units

            
    def __repr__(self):
        return f"Experiment \n Strain: {self.strain},\n Condition: {self.condition}\n N = {len(self.samples)}."
    

    def __add__(self, other):
        assert self.strain == other.strain, "Strains don't match."
        assert self.condition == other.condition, "conditions don't match"
        return Experiment(self.strain, self.condition, self.scale, self.fps, samples = [*self.samples, *other.samples])
    

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, key):
        if isinstance(key, slice):
            # do your handling for a slice object:
            samples = self.samples[key.start:key.stop:key.step]
            return Experiment(self.strain, self.condition, self.scale, self.fps, samples = samples.copy())
        elif isinstance( key, int ):
            if key < 0 : #Handle negative indices
                key += len( self )
            if key < 0 or key >= len(self) :
                raise IndexError(f"The index ({key}) is out of range.")
            sample = self.samples[key]
            return Experiment(self.strain, self.condition, self.scale, self.fps, samples = [sample].copy())
        else:
            raise TypeError("Invalid argument type.")
    ######################################
    #
    #  Data loading
    #
    #######################################
    
    def load_data(self, path, columns = None, append = True, nmax = None, filterword = "", units = None, **kwargs):
        """
        Load all results files from a folder. 
        input:
            path: location of pharaglow results files
            columns: which colums to load.
            append (bool): append to existing samples. If False, start with an empty experiment.
            nmax: maximum number of samples to load
            units: units of columns in loaded files
        """
        # Track if columns were explicitly provided 
        user_specified_columns = columns is not None
        # when calls load_data(path) → columns = None → user_specified_columns = False
        # when calls load_data(path, columns=['x','y']) → columns = [...] → user_specified_columns = True
        
        if columns is None:
             columns = ['x', 'y', 'frame', 'pumps'] # default list when no list specified 
        
        # Warn if using only minimal columns 
        if set(columns) <= {'x', 'y', 'frame'}:
            warnings.warn(
                f"Using minimal column set {sorted(columns)}. "
                "Consider adding additional columns for full analysis.",
                UserWarning
            )
            
        # unit definitions
        if units is None:
            # Use the default UNITS dictionary
            self.units = UNITS.copy()
            # Warn if user specified columns but not units
            if user_specified_columns:
                warnings.warn(
                    "Columns were specified without explicit units. "
                    "Using default units from UNITS dictionary. "
                    "To suppress this warning, provide units as a dict or YAML file path.",
                    UserWarning
                )
        elif isinstance(units, str) or isinstance(units, Path):
            with open(units) as stream:
                try:
                    self.units = yaml.safe_load(stream)
                except yaml.YAMLError as exc:
                    print(exc)
        elif isinstance(units, dict):
            self.units = units
        else:
            raise RuntimeError('units must be None, a dict, or a path to a YAML config file!')
        # check if we have units for all columns
        if not set(columns)<=set(self.units):
            raise IndexError(f"Units are not specified for all columns {set(self.units)}.")
        # add the units for scale and fps
        self.units['space_units'] = self.space_units
        self.units['time_units'] = self.time_units
        
        # load stuff
        if nmax == None:
            nmax = np.inf
        if not append:
            self.samples = []
        j = 0
        for fn in os.listdir(path):
            file = os.path.join(path,fn)
            if j >= nmax:
                break
            if os.path.isfile(file) and filterword in fn and fn.endswith('.json'):
                self.samples.append(Worm(file, columns, self.fps, self.scale, self.units, **kwargs))
                j += 1
    
    
    def save_wcon(self, filepath, columns = None, tag = '@INF'):
        """ Save the Experiment as a valid wcon json file.

        Args:
            file_path (str) : Path to the output file. should end in .wcon
            
        Returns:
            -
        """
        tmp_dict = {}
        ### create correct structure for wcon
            # units for each column
        tmp_dict['units'] = {}
       
        # experimental metadata
        try:
            tmp_dict['metadata'] = self.experiment_metadata
        except KeyError:
            print('Please define the experimental metadata first by providing a dictionary to self.define_metadata.')
        
        # get the scaled coordinates
        self.calculate_property('locations')
        # update units
        self.update_units()
        # store the fps and scale information
        tmp_dict['metadata']['@imaging_setup'] = {}
        tmp_dict['metadata']['@imaging_setup']['scale'] = self.scale
        tmp_dict['metadata']['@imaging_setup']['fps'] = self.fps
        tmp_dict['units']['scale'] = f"{self.units['space_units']}/px"
        tmp_dict['units']['fps'] = f"1/{self.units['time_units']}"
        # add units to some metadata entries
        for key in tmp_dict['metadata']:
            if key in self.units.keys():
                tmp_dict['units'][key] = self.units[key]
        
        # get each worm as data    
        tmp_dict['data'] = []
        for worm in self.samples:
            data = {}
            worm.create_ID()
            data['id'] = worm.id
            if columns is None:
                tmp = worm.data.to_dict(orient='list')
                columns = tmp.keys()
            else:
                tmp = worm.data.filter(columns)
                tmp = tmp.to_dict(orient='list')
            # if index is accicentally left in the columns - remove it
            tmp.pop('index', '')
            # extract the t,x,y columns
            for wcon_key,inf_key in zip(['x', 'y', 't'],['x_scaled', 'y_scaled', 'time']):
                data[wcon_key] = worm.data[inf_key].values.tolist()
                tmp_dict['units'][wcon_key] = worm.units[inf_key]
                # pop duplicate keys
                tmp.pop(inf_key, '')
            # add a custom tag in front of the custom metrics
            data[tag] = tmp
            # add the units for our custom tags
            for key in tmp:
                tmp_dict['units'][key] = self.units[key]
            tmp_dict['data'].append(data)
            # add centerlines
            if any([key in columns for key in ['Centerline' ,'centerline']]):
                data[tag]['centerline_x'] = worm.centerline_scaled[:,:,1].tolist()
                data[tag]['centerline_y'] = worm.centerline_scaled[:,:,0].tolist()
                tmp_dict['units']['centerline_x'] = worm.units['centerline_scaled']
                tmp_dict['units']['centerline_y'] = worm.units['centerline_scaled']
        # write data
        with open(filepath, 'w', encoding='utf-8') as f:
            # dump as json
            json.dump(tmp_dict, f, ensure_ascii=False, indent=4)
            
            
    def define_metadata(self, info, units = None):
        """Add experimental metadata e.g., temperature, detailed conditions,...."""
        
        if isinstance(info, dict):
            assert info['strain'] == self.strain, 'Ensure that the strain definition in the metadata matches the strain name in the Experiment!'
            self.experiment_metadata = info
            if isinstance(units, dict):
                z = {**self.units, **units}
                self.units = z
        else:
            raise RuntimeError('info should be a dictionary.')
    
    def create_IDs(self):
        """create a unique ID matching raw data for each sample"""
        for worm in self.samples:
            worm.create_ID()
    
    def update_units(self):
        """Synchronize units with all worm sample units."""
        old_units = self.units
        for worm in self.samples:
            old_units = {**old_units, **worm.units}
        self.units = old_units
            
        
    def load_stimulus(self, filename):
        """load a stimulus file"""
        #TODO test and adapt
        self.stimulus = np.loadtxt(filename)
    
    
    def add_column(self, key, values, overwrite = True):
        """add a column of data to each worm. 
            key: name of new (or existing) column
            values: Array or List containing the values added to each sample in the experiment.
            overwrite: overwrite column if it exists.
        """
        assert len(values) == len(self.samples), f'Number of values provided {len(values)} does not match the number of samples in the experiment {len(self.samples)}.'
        for n, worm in enumerate(self.samples):
            worm.add_column(key, values[n], overwrite)


    def align_data(self, timepoints, tau_before, tau_after, key = None, column_align = 'frame'):
        """calculate aligned data for all worms"""
        for worm in self.samples:
            worm.multi_align(timepoints, tau_before, tau_after, key = key, column_align = column_align)


    def calculate_property(self, name, **kwargs):
        """calculate additional properties on the whole dataset, calling functions for each worm."""

        # save metadata
        key = kwargs.pop('key', 'default')
        self.metadata[f"{name}_{key}"] = {}
        for keyword in kwargs:
            self.metadata[f"{name}_{key}"][keyword] = kwargs[keyword]
#         # run function
        if key is not 'default':
            kwargs['key'] = key
        for worm in self.samples:
             worm.calculate_property(name, **kwargs)

   
    ######################################
    #
    #   get/set attributes
    #
    #######################################
    def set_color(self, color):
        """sets the color used for plotting this experiment. Can be a defined color string or hex-code.
            Anything that matplotlib understands is valid.
        """
        self.color = color


    def get_sample(self, N):
        """get a sample of the experiment.
            Returns a Worm() object
        """
        return self.samples[N]

    def get_sample_metric(self, key, metric = None, filterfunction = None, axis = 1, ignore_index = True):
        """ Metrics across samples as a function of time (axis = 1) or averaged over time a function of samples (axis = 0).
            metric: one of 'sum', 'mean', 'std', 'N', 'sem', 'median', rate', 'collapse', 'max' or 'min'
            filterfunction should be a callable that will be applied to each sample and evaluate to True or False for each aligned dataset.
            axis: axis = 1 - returns the sample-averaged timeseries of the data, axis = 0 returns the time-averaged/metric of each sample in the data.
            ignore_index: TODO
        """
        tmp = []
        for worm in self.samples:
            tmp.append(worm.get_data(key))
        # set index to frame
        tmp = pd.concat(tmp, axis = 1, sort=True)
        if filterfunction is not None:
            filtercondition = tmp.apply(filterfunction)
            tmp = tmp.loc[:,filtercondition]
        tmp.columns = [f'{x}_{i}' for i, x in enumerate(tmp.columns, 1)]
        if metric == None:
            return tmp
        if metric == "sum":
            return tmp.sum(axis = axis)
        if metric == "mean":
            return tmp.mean(axis = axis)
        if metric == "mean_%":
            return tmp.mean(axis=1)*100
        if metric == "std":
            return tmp.std(axis = axis)
        if metric == "N":
            return tmp.count(axis = axis)
        if metric == "sem":
            return tmp.std(axis = axis)/self.get_sample_metric(key, 'N', axis=axis)**0.5
        if metric == "median":
            return tmp.median(axis = axis)
        if metric == "rate":
            return tmp.sum(axis=axis)/tmp.count(axis=axis)*self.fps
        if metric == "collapse":
            return pd.DataFrame(tmp.values.ravel(), columns = [key])
        if metric == 'max':
            return tmp.max(axis = axis)
        if metric == 'min':
            return tmp.min(axis = axis)
        else:
            raise Exception("Metric not implemented, choose one of 'mean','mean_%', 'std', 'sem', 'sum', 'collapse', 'median', 'max', 'min' or 'N'")


    def get_aligned_sample_metric(self, key, metric_sample = None, metric_timepoints =  'mean', filterfunction = None, axis = 1):
        """ Metrics across samples. 
            metric_sample is the function applied across the worms in this experiment.
            metric_timepoints is the function applied across stimuli (this is trivial if only one time alignment existed.)
            filterfunction should be a callable that will be applied to each sample and evaluate to True or False for each aligned dataset.
            e.g. get_aligned_sample_metric('velocity', 'mean', 'mean') would return the mean(mean(v, N_timepoints), N_worms).

        """
        tmp = []
        for worm in self.samples:
            # if we give no metric, all stimuli will be attached.
            if metric_timepoints == None:
                tmp.append(worm.get_data_aligned(key))
                #tmp = pd.concat(tmp, axis = 1)
            else:
                tmp.append(worm.get_aligned_metric(key, metric_timepoints))
        tmp = pd.concat(tmp, axis = 1)
        tmp.columns = [f'{x}_{i}' for i, x in enumerate(tmp.columns, 1)]
        if filterfunction is not None:
            filtercondition = tmp.apply(filterfunction)
            tmp = tmp.loc[:,filtercondition]
        if metric_sample == None:
            return tmp
        if metric_sample == "sum":
            return tmp.sum(axis = axis)
        if metric_sample == "mean":
            return tmp.mean(axis = axis)
        if metric_sample == "mean_%":
            return tmp.mean(axis = axis)*100
        if metric_sample == "std":
            return tmp.std(axis = axis)
        if metric_sample == "N":
            return tmp.count(axis = axis)
        if metric_sample == "median":
            return tmp.median(axis = axis)
        if metric_sample == "sem":
            return tmp.std(axis = axis)/self.get_aligned_sample_metric(key, 'N', axis = axis)**0.5
        if metric_sample == "collapse":
            return pd.DataFrame(tmp.values.ravel(), columns=[key])
        if metric_sample == 'max':
            return tmp.max(axis = axis)
        if metric_sample == 'min':
            return tmp.min(axis = axis)
        else:
            raise Exception("Metric not implemented, choose one of 'mean', 'mean_%', 'std', 'sem', 'sum', 'collapse', 'median', 'max', 'min' or 'N'")
    

    def get_events(self, events = 'pump_events' ,unit = None, aligned = False):
        """ get peak locations for all samples.
            events: column with binary entries. 
            unit: column of data at which to evaluate e.g. 'time'
            return: list of peaks.
        """
        tmp = []
        for worm in self.samples:
            tmp.append(worm.get_events(events, unit, aligned))
        return tmp
    ######################################
    #
    #   Plotting functions
    #
    #######################################

    def plot(self, ax, keys, metric, metric_sample = None, plot_type = 'line', metric_error = None, filterfunction = None, aligned = False, axis = 1,  apply_to_x = True, **kwargs):
        """plot the experiment.
            keys: list of strings or single string, column of data in the Worm object. Will use 'time' for x if using a 2d plot style., ...
            metric_sample: is the function applied across the worms in this experiment ; can be a single None OR a single string variable (='mean') OR a list with metric_sample_x and metric_sample_y (=[''mean','N'])
            metric: is the function applied across time (or stimuli for aligned data)
            filterfunction should be a callable that will be applied to each sample and evaluate to True or False for each aligned dataset.
            aligned: Use self.samples.aligned_data or self.samples.data
            ax: either matplotlib axis object or list of axes
            metric: if true, plot the sample metric of each key
            axis: only used if aligned = True: axis = 1 metric across columns -> result is a timeseries axis = 0 metric across rows -> results is one for each sample/worm or stimulus.
            apply_to_x: apply the same metric to the x-axis/first key. Set to False if you want to plot e.g., a timeseries
        """
        if isinstance(keys, list) or isinstance(keys, tuple):
            key_x, key_y = keys[:2]
        elif isinstance(keys, str):
            key_y = keys
            key_x = 'time'
        else:
            raise ValueError(f'The entry for keys {keys} is not valid.')
        xerr = None
        yerr = None
        
        
        if metric_sample == None:
            metric_sample_x = None
            metric_sample_y = None
        elif isinstance(metric_sample, str):
            metric_sample_x = metric_sample
            metric_sample_y = metric_sample
        else:
            metric_sample_x = metric_sample[0]
            metric_sample_y = metric_sample[1]
            
        if aligned:
            # time is not meaningful, choose a different key
            if key_x == 'time':

                key_x = 'time_align'
            if apply_to_x:
                x = self.get_aligned_sample_metric(key_x, metric_sample, metric, filterfunction, axis)
            else:
                if metric_sample is not None:
                    x = self.get_aligned_sample_metric(key_x, 'mean', None, filterfunction, axis)
                elif metric is not None:
                    x = self.get_aligned_sample_metric(key_x, None, 'mean', filterfunction, axis)
                
            
            y = self.get_aligned_sample_metric(key_y, metric_sample, metric, filterfunction, axis)
            

                
            if metric_error is not None:
                xerr = self.get_aligned_sample_metric(key_x, metric_error, metric, filterfunction, axis)
                yerr = self.get_aligned_sample_metric(key_y, metric_error, metric, filterfunction, axis)
        else:
            if metric == None and metric_sample == None:
                # return the full joined data array of all samples
                x = self.get_sample_metric(key_x, None, filterfunction)
                y = self.get_sample_metric(key_y, None, filterfunction)
            elif metric_sample == None:
                # return metric for each worm - result will be Nsamples long
                x = self.get_sample_metric(key_x, metric, filterfunction, axis = 0)
                y = self.get_sample_metric(key_y, metric, filterfunction, axis = 0)
                if metric_error is not None:
                    xerr = self.get_sample_metric(key_x, metric_error, filterfunction, axis = 0)
                    yerr = self.get_sample_metric(key_y, metric_error, filterfunction, axis = 0)
            elif metric == None:
                # return the metric across each trajectory(Worm) - result will be an average across samples
                warnings.warn('This option keeps the dataframe index while applying the sample metric which is rarely meaningful. You probably want to align all datasets to their t=0 and rerun with the aligned option.')
                x = self.get_sample_metric(key_x, metric_sample_x, filterfunction, axis = 1)
                y = self.get_sample_metric(key_y, metric_sample_y, filterfunction, axis = 1)
                if metric_error is not None:
                    xerr = self.get_sample_metric(key_x, metric_error, filterfunction, axis = 1)
                    yerr = self.get_sample_metric(key_y, metric_error, filterfunction, axis = 1)
            else:
                warnings.warn('Pick either a sample or timeseries metric. Both reducing operations are not meaningful to plot.')
                return None, None, None
        # check if user overrode color keyword
        kwargs['color'] =  kwargs.pop('color', self.color)
        
        if plot_type == 'line':
            plot = _lineplot(x ,y, yerr, ax, **kwargs)

        elif plot_type == 'histogram':
            plot = _hist(y, ax , **kwargs)

        elif plot_type == 'scatter':
            plot = _scatter(x, y, None, yerr, ax, density = False, **kwargs)

        elif plot_type == 'density':
            plot = _scatter(x, y, xerr, yerr,  ax, density = True, **kwargs)

        elif plot_type == 'xy_error_scatter':
            plot = _scatter(x, y, xerr, yerr, ax, density = False, **kwargs)

        elif plot_type == 'raster' or plot_type == 'heatmap':
            color = kwargs.pop('color')
            plot = _heatmap(x, y, ax, **kwargs)

        elif plot_type == 'bar':
            print("Don't use bar plots! Really? beautiful boxplots await with plot_type = 'box'")
            loc = kwargs.pop('loc', 0)
            plot = ax.bar(loc, y, label = self.strain, **kwargs)

        # elif plot_type == 'box':
        #     loc = kwargs.pop('loc', 0)
        #     color = kwargs.pop('color', self.color)
        #     plot = style.scatterBoxplot(ax, [loc], [y], [color], [self.strain], **kwargs)
            
        elif plot_type == 'box':
            loc = kwargs.pop('loc', 0)
            color = kwargs.pop('color', self.color)
            lbls = kwargs.pop('lbls', self.strain)
            plot = style.scatterBoxplot(ax,  x_data = [loc], y_data = [y], clrs = [color], lbls = [lbls], **kwargs)
           
                    
        else:
             raise NotImplementedError("plot_type not implemented, choose one of 'line', 'histogram', 'scatter', 'density', 'bar', 'box'.")
        return plot, x, y
