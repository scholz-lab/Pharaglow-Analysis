import os
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from . import style
from . import tools
from .tools import PickleDumpLoadMixin

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
    def __init__(self, filename, columns,fps, scale, particle_index = None, **kwargs):
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
        # load data
        print('Reading', filename)
        self._load(filename, columns, fps, scale, **kwargs)


    def _load(self, filename, columns, fps, scale, **kwargs):
        """load data."""
        traj = pd.read_json(filename, orient='split')
        # drop all columns except the ones we want - but keep the minimal values
        traj = traj.filter(columns)
        # extract the centerlines and other non-scalar values into an array instead
        if 'Centerline' in columns:
            self.centerline = np.array([np.array(cl) for cl in traj['Centerline']])
        if 'Straightened' in columns:
            self.images = np.array([np.array(im) for im in traj['Straightened']])
        traj.drop(['Centerline', 'Straightened'], errors = 'ignore')
        # velocity and real time
        traj['time'] = traj['frame']/fps
           
        try:
            traj['velocity'] = np.sqrt((traj['x'].diff()**2+traj['y'].diff()**2))/traj['frame'].diff()*scale*fps
        except KeyError:
            print('Velocity calculation failed. Continuing.')
        self.data = traj
        try:
        # pumping related data
            if "w_bg" not in kwargs.keys():
                kwargs["w_bg"] = 10
                print(f'Setting Background windows to {kwargs["w_bg"]} for pump extraction')
            if "w_sm" not in kwargs.keys():
                kwargs["w_sm"] = 2
                print(f'Setting smoothing windows to {kwargs["w_sm"]} for pump extraction')
            if "sensitivity" not in kwargs.keys():
                kwargs["sensitivity"] = 0.9
                print(f'Setting sensitivity to {kwargs["sensitivity"]} for pump extraction')
            if "min_distance" not in kwargs.keys():
                kwargs["min_distance"] = 4
                print(f'Setting peak distance to {kwargs["min_distance"]} for pump extraction')
            self.calculate_pumps(kwargs["w_bg"], kwargs["w_sm"], kwargs["min_distance"],  kwargs["sensitivity"])
        except Exception:
             print('Pumping extraction failed. Try with different parameters.')
             self.flag = True
             self.data = traj
        finally: 
            # ensure numerical index
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
        else:
            raise Exception("Metric not implemented, choose one of 'mean','median', 'std', 'sem' , 'sum', 'rate', or 'N'")
    
    
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

        else:
            raise Exception("Metric not implemented, choose one of 'mean', 'median', 'std', 'sem', 'sum,' or 'N'")


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
    def calculate_property(self, name, **kwargs):
        """calculate additional properties on the whole dataset, calling functions for each worm."""

        funcs = {"reversals": self.calculate_reversals,
                "count_rate": self.calculate_count_rate,
                "smoothed": self.calculate_smoothed,
                "pumps": self.calculate_pumps,
                "nose_speed": self.calculate_nose_speed,
                "reversals_nose": self.calculate_reversals_nose
        }
        if name == 'help':
            print(funcs)
            return
        # run function
        funcs[name](**kwargs)


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
                dset[f'{key}_smooth'] = dset[key].rolling(window, **kwargs).mean()
        else:
            self.data[f'{key}_smooth'] = self.data[key].rolling(window, **kwargs).mean()


    def calculate_pumps(self, w_bg, w_sm, min_distance,  sensitivity, key = 'pumps', **kwargs):
        """using a pump trace, get additional pumping metrics."""
        # remove outliers
        self.data['pump_clean'] = tools.hampel(self.data[key], w_bg*30)
        self.data['pump_clean'],_ = tools.preprocess(self.data['pump_clean'], w_bg, w_sm)
        # deal with heights for the expected peaks
        ### here we make the heights sensible: threshold between median and maximum of trace
        h = np.linspace(self.data['pump_clean'].median(), self.data['pump_clean'].max(), 50)
        heights = kwargs.pop('heights', h)
        peaks, _,_  = tools.find_pumps(self.data['pump_clean'], min_distance=min_distance,  sensitivity=sensitivity, heights = heights)
        if len(peaks)>0:
            # add interpolated pumping rate to dataframe
            self.data['rate'] = np.interp(np.arange(len(self.data)), peaks[:-1], self.fps/np.diff(peaks))
            # # get a binary trace where pumps are 1 and non-pumps are 0
            self.data['pump_events'] = 0
            self.data.loc[peaks,['pump_events']] = 1
        else:
            self.data['rate'] = 0
            self.data['pump_events'] = 0

        
    def calculate_count_rate(self, window, **kwargs):
        """Add a column 'count_rate' to self.data. Calculate a pumping rate based on number of counts of pumps in a window. 
        window is in frame. Result will be in Hz."""
        kwargs['center'] =  kwargs.pop('center', True)
        kwargs['min_periods'] =  kwargs.pop('min_periods', 1)
        self.data['count_rate'] = self.data['pump_events'].rolling(window, **kwargs).sum()/window*self.fps


    def calculate_reversals(self, animal_size, angle_threshold):
        """Adaptation of the Hardaker's method to detect reversal event. 
        A single worm's centroid trajectory is re-sampled with a distance interval equivalent to 1/10 
        of the worm's length (100um) and then reversals are calculated from turning angles.
        Inputs:
            animal_size: animal size in um.
            angle_threshold (degree): what defines a turn 
        Output: None, but adds a column 'reversals' to  self.data.
        """
        # resampling
        # Calculate the distance cover by the centroid of the worm between two frames um
        distance = np.cumsum(self.data['velocity'])*self.data['time'].diff()
        # find the maximum number of worm lengths we have travelled
        maxlen = distance.max()/animal_size
        # make list of levels that are multiples of animal size
        levels = np.arange(animal_size, maxlen*animal_size, animal_size)
        # Find the indices where the distance is equal or the closest to the pixel interval by repeatedly subtracting the levels
        indices = []
        for level in levels:
            idx = distance.sub(level).abs().idxmin()
            indices.append(idx)
        # create a downsampled trajectory from these indices
        traj_Resampled = self.data.loc[indices, ['x', 'y']].diff()
        # we ignore the index here for the shifted data
        traj_Resampled[['x1', 'y1']] = traj_Resampled.shift(1).fillna(0)
        # use the dot product to calculate the andle
        def angle(row):
            v1 = [row.x, row.y]
            v2 = [row.x1, row.y1]
            return np.degrees(np.arccos(np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)))
        traj_Resampled['angle'] = traj_Resampled.apply(lambda row: angle(row), axis =1)
        rev = traj_Resampled.index[traj_Resampled.angle>=angle_threshold]
        self.data['reversals'] = 0
        self.data.loc[rev,'reversals'] = 1
     

    def calculate_reversals_nose(self, dt =1, angle_threshold = 150):
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
        nose = cl_new[:,0,:]
        #calculate directions - in the lab frame! and with dt
        v_nose = nose[dt:]-nose[:-dt]
        #v_cms = cms[dt:]-cms[:-dt]
        # tangent of the worm = 'heading'
        heading = cl_new[:,0,]-cl_new[:,nPts//2,]
        
        # angle relative to cms motion
        angle = np.array([np.arccos(np.dot(v1, v2)/(np.linalg.norm(v2)*np.linalg.norm(v1)))*180/np.pi for \
                          (v1, v2) in zip(v_nose, heading)])
        angle[np.isnan(angle)] = 0
        # determine when angle is over threshold
        rev = angle > angle_threshold
        rev = np.append(rev, np.nan)
        self.add_column('reversals', rev, overwrite = True)
    

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
        deltat = self.data.time[dt:]-self.data.time[:-dt]
        v_nose_abs = np.sqrt(np.sum((v_nose)**2, axis = 1))/deltat*self.scale#*self.fps
        v_cms_abs = np.sqrt(np.sum((v_cms)**2, axis = 1))/deltat*self.scale#*self.fps
        # add back the missing item from difference
        v_nose_abs = np.append(v_nose_abs, np.nan)
        v_cms_abs = np.append(v_cms_abs, np.nan)
        # add to data
        self.add_column('nose_speed', v_nose_abs, overwrite = True)
        self.add_column('cms_speed', v_cms_abs, overwrite = True)
 

    def align(self, timepoint,  tau_before, tau_after, key = None, column_align = 'frame'):
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
        tstart, tend = timepoint -tau_before, timepoint+tau_after
        frames = np.arange(tstart, tend+1)
        tmp = self.data[self.data[column_align].isin(frames)].loc[:,key]
        # fill missing data
        tmp = tmp.set_index(column_align)
        tmp = tmp.reindex(pd.Index(frames))
        tmp.index = pd.Index(np.arange(-tau_before, tau_after+1))
        tmp['time_align'] = tmp.index.values/self.fps
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
    def __init__(self, strain, condition, scale, fps, samples = None, color = None):
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
            return Experiment(self.strain, self.condition, self.scale, self.fps, samples = [sample.copy()])
        else:
            raise TypeError("Invalid argument type.")
    ######################################
    #
    #  Data loading
    #
    #######################################
    def load_data(self, path, columns = ['x', 'y', 'frame', 'pumps'], append = True, nmax = None, filterword = None, **kwargs):
        """load all results files from a folder. 
            Inputs:
                path: location of pharaglow results files
                columns: required columns for analysis.
            Params: 
                append: append to existing samples. If False, start with an empty experiment.
        """
        
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
                self.samples.append(Worm(file, columns, self.fps, self.scale, **kwargs))
                j += 1
                

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
        self.metadata[name] = {}
        for keyword in kwargs:
             self.metadata[name][keyword] = kwargs[keyword]
        # run function
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


    def get_sample_metric(self, key, metric = None, filterfunction = None, axis = 1, ignore_index = False):
        """ Metrics across samples as a function of time (axis=1) or averaged over time a function of samples (axis = 0).
            metric: one of 'mean', 'std', 'N' or 'sem.'
            filterfunction should be a callable that will be applied to each sample and evaluate to True or False for each aligned dataset.
            axis: axis = 1 - returns the sample-averaged timeseries of the data, axis = 0 returns the time-averaged/metric of each sample in the data.

        """
        tmp = []
        for worm in self.samples:
            tmp.append(worm.get_data(key))
        # set index to frame
        tmp = pd.concat(tmp, axis = 1, ignore_index = ignore_index)
        if filterfunction is not None:
            filtercondition = tmp.apply(filterfunction)
            tmp = tmp.loc[:,filtercondition]
        tmp.columns = [f'{x}_{i}' for i, x in enumerate(tmp.columns, 1)]
        if metric ==None:
            return tmp
        if metric == "sum":
            return tmp.sum(axis = axis)
        if metric == "mean":
            return tmp.mean(axis = axis)
        if metric == "std":
            return tmp.std(axis = axis)
        if metric == "N":
            return tmp.count(axis = axis)
        if metric == "sem":
           return tmp.std(axis = axis)/self.get_sample_metric(key, 'N', axis=axis)**0.5
        if metric == "rate":
            return tmp.sum(axis=axis)/tmp.count(axis=axis)*self.fps
        if metric == "collapse":
            return pd.DataFrame(tmp.values.ravel())
        else:
            raise Exception("Metric not implemented, choose one of 'mean', 'std', 'sem', 'sum', 'collapse' or 'N'")


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
        if metric_sample ==None:
            return tmp
        if metric_sample == "sum":
            return tmp.sum(axis = axis)
        if metric_sample == "mean":
            return tmp.mean(axis = axis)
        if metric_sample == "std":
            return tmp.std(axis = axis)
        if metric_sample == "N":
            return tmp.count(axis = axis)
        if metric_sample == "sem":
           return tmp.std(axis = axis)/self.get_aligned_sample_metric(key, 'N', axis = axis)**0.5
        if metric_sample == "collapse":
            return pd.DataFrame(tmp.values.ravel())
        else:
            raise Exception("Metric not implemented, choose one of 'mean', 'std', 'sem', 'sum', 'collapse' or 'N'")
    

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

    def plot(self, ax, keys, metric, metric_sample = None, plot_type = 'line', metric_error = None, filterfunction = None, aligned = False, axis = 1,  **kwargs):
        """plot the experiment.
            keys: list of strings or single string, column of data in the Worm object. Will use 'time' for x if using a 2d plot style., ...
            metric_sample: is the function applied across the worms in this experiment.
            metric: is the function applied across time (or stimuli for aligned data)
            filterfunction should be a callable that will be applied to each sample and evaluate to True or False for each aligned dataset.
            aligned: Use self.samples.aligned_data or self.samples.data
            ax: either matplotlib axis object or list of axes
            metric: if true, plot the sample metric of each key
            axis: only used if aligned = True: axis = 1 metric across columns -> result is a timeseries axis = 0 metric across rows -> results is one for each sample/worm or stimulus.
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
        if aligned:
            # time is not menaingful, choose a different key
            if key_x == 'time':
                key_x = 'time_align'
            x = self.get_aligned_sample_metric(key_x, metric_sample, metric, filterfunction, axis)
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
                x = self.get_sample_metric(key_x, metric_sample, filterfunction, axis = 1)
                y = self.get_sample_metric(key_y, metric_sample, filterfunction, axis = 1)
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

        elif plot_type == 'box':
            loc = kwargs.pop('loc', 0)
            color = kwargs.pop('color', self.color)
            plot = style.scatterBoxplot(ax, [loc], [y], [color], [self.strain], **kwargs)
        else:
             raise NotImplementedError("plot_type not implemented, choose one of 'line', 'histogram', 'scatter', 'density', 'bar', 'box'.")
        return plot, x, y
