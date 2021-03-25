import os
import pickle
import warnings


import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from . import style
from .tools import PickleDumpLoadMixin
from pharaglow import io, extract


class Worm(PickleDumpLoadMixin):
    """class to contain data from a single pharaglow result."""
    def __init__(self, filename, columns,fps, scale, **kwargs):
        """initialize object and load a pharaglow results file."""
        self.fps = fps
        self.scale = scale
        self.flag = False
        print('Reading', filename)
        # keep some metadata
        self.experiment = os.path.basename(filename)
        self.particle_index = int(os.path.splitext(self.experiment)[0].split('_')[-1])
        # load data
        self._load(filename, columns, fps, scale)


    def _load(self, filename, columns, fps, scale, **kwargs):
        """load data."""
        traj = io.load(filename, orient='split')
        # drop all columns except the ones we want - but keep the minimal values
        traj = traj.filter(columns)
        # velocity and real time
        traj['time'] = traj['frame']/fps
        #print(traj.info())
        traj['velocity'] = np.sqrt((traj['x'].diff()**2+traj['y'].diff()**2))/traj['frame'].diff()*scale*fps
        # pumping related data
        try:
            traj['pump_clean'] = extract.preprocess(traj['pumps'])
            peaks, _,_  = extract.find_pumps(traj['pumps_clean'], **kwargs)
            # reset peaks to match frame
            peaks += np.min(traj.frame)
            # add interpolated pumping rate to dataframe
            traj['rate'] = np.interp(traj['frame'], peaks[:-1], fps/np.diff(peaks))
            # # get a binary trace where pumps are 1 and non-pumps are 0
            traj['pump_events'] = 0
            traj.loc[peaks,['pump_events']] = 1
        except Exception:
            print('Pumping extraction failed. Try with different parameters.')
            self.flag = True
            self.traj = []
        self.data = traj
    

    def __repr__(self):
        return f"Worm \n with underlying data: {self.data.describe()}"


    def __len__(self):
        return len(self.data)
    ######################################
    #
    #   get/set attributes
    #
    #######################################

    def get_metric(self, key, metric):
        """return metrics of a data column given by key."""
        assert key in self.data.columns, f'The key {key} does not exist in the data.'
        if metric == "mean":
            return self.data[key].mean()
        
        if metric == "std":
            return self.data[key].std()
        
        if metric == "N":
            return self.data[key].size
        
        if metric == "sem":
           return self.data[key].std()/np.sqrt(self.data[key].size)
        else:
            raise Exception("Metric not implemented, choose one of 'mean', 'std', 'sem' or 'N'")
    
    
    def get_aligned_metric(self, key, metric):
        """get averages across timepoints for a single worm. eg. average across multiple stimuli. 
        Requires multi_align(self) to be run."""
        assert len(self.aligned_data )>0, 'Please run Worm.align() or Worm.multi_align() first!'
        assert key in self.data.columns, f'The key {key} does not exist in the data.'
        tmp = pd.concat([data[key] for data in self.aligned_data], axis = 1)
        if metric == "mean":
            return tmp.mean(axis = 1)
        if metric == "std":
            return tmp.std(axis = 1)
        if metric == "N":
            return tmp.count(axis =1)
        if metric == "sem":
           return tmp.std(axis = 1)/np.sqrt(tmp.count(axis=1))
        else:
            raise Exception("Metric not implemented, choose one of 'mean', 'std', 'sem' or 'N'")


    def get_data(self, key = None):
        """return a column of data or the whole pandas dataframe."""
        if key == None:
            return self.data
        else:
            assert key in self.data.columns, f'The key {key} does not exist in the data.'
            return self.data[key]


    def get_data_aligned(self, index, key = None):
        """return a column of aigned data or the whole aligned pandas dataframe at a specific timepoints."""
        if key == None:
            return self.data
        else:
            assert key in self.data.columns, f'The key {key} does not exist in the data.'
            return self.data[key]
            
    

    def calculate_reversals(self, animal_size, angle_treshold):
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
        rev = traj_Resampled.index[traj_Resampled.angle>=angle_treshold]
        self.data['reversals'] = 0
        self.data.loc[rev,'reversals'] = 1
    

    def align(self, timepoint,  tau_before, tau_after, key = None):
        """align to a timepoint.
         Inputs:
                timepoint: time to align to in frames
                tau_before: number of frames before timepoint
                tau_after: number of frames after timepoint
                key is a string or list of strings as column names.
        Output: creates a list of aligned dataframes centered around the timepoint 
        """
        if key == None:
            key = self.data.columns
        tstart, tend = timepoint -tau_before, timepoint+tau_after
        tmp = self.data.loc[tstart:tend, key]
        tmp = tmp.reindex(pd.Index(np.arange(tstart, tend)))
        tmp.index = pd.Index(np.arange(-tau_before, tau_after))
        return tmp
    

    def multi_align(self, timepoints, tau_before, tau_after, key = None):
        """align to multiple timepoints.
         Inputs:
                timepoints: list of timepoints to align to in frames
                tau_before: number of frames before timepoint
                tau_after: number of frames after timepoint
                key is a string or list of strings as column names.
        Output: creates a list of aligned dataframes centered around the timepoint 
        """
        self.aligned_data = []
        self.timepoints = timepoints
        if key == None:
            key = self.data.columns
        for timepoint in self.timepoints:
            tmp = self.align(timepoint,  tau_before, tau_after, key = None)
            self.aligned_data.append(tmp)


    def calculate_count_rate(self, window):
        """Add a column 'count_rate' to self.data. Calculate a pumping rate based on number of counts of pumps in a window. 
        window is in frame. Result will be in Hz."""
        self.data['count_rate'] = self.data['pump_events'].rolling(window, center=True, min_periods=1).sum()/window*self.fps

    
class Experiment(PickleDumpLoadMixin):
    """Wrapper class which is a container for individual worms."""
    # class attributes
    def __init__(self, strain, condition, scale, fps, samples = None, color = None):
        self.strain = strain
        self.condition = condition
        self.scale = scale
        self.fps = fps

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
            return Experiment(self.strain, self.condition, self.scale, self.fps, samples = samples)
        elif isinstance( key, int ):
            if key < 0 : #Handle negative indices
                key += len( self )
            if key < 0 or key >= len(self) :
                raise IndexError(f"The index ({key}) is out of range.")
            sample = self.samples[key]
            return Experiment(self.strain, self.condition, self.scale, self.fps, samples = [sample])
        else:
            raise TypeError("Invalid argument type.")
    ######################################
    #
    #  Data loading
    #
    #######################################
    def load_data(self, path, columns = ['x', 'y', 'frame', 'pumps'], append = True, nmax = None):
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
            if os.path.isfile(file) and 'results_' in fn and fn.endswith('.json'):
                self.samples.append(Worm(file, columns, self.fps, self.scale))
                j += 1
                

    def load_stimulus(self, filename):
        """load a stimulus file"""
        #TODO test and adapt
        self.stimulus = np.loadtxt(filename)
    

    def align_data(self, timepoints, tau_before, tau_after, key = None):
        """calculate aligned data for all worms"""
        for worm in self.samples:
            worm.multi_align(timepoints, tau_before, tau_after, key = None)


    def calculate_reversals(self, animal_size, angle_treshold):
        """calculate the reversals for each worm"""
        for worm in self.samples:
            worm.calculate_reversals(animal_size, angle_treshold)
    ######################################
    #
    #   get/set attributes
    #
    #######################################
    def set_color(self, color):
        """sets the color used for plotting this experiment. Can be a defined color string or hex-code.
            Anything that matlab understands is valid.
        """
        self.color = color


    def get_sample(self, N):
        """get a sample of the experiment.
            Returns a Worm() object
        """
        return self.samples[N]


    def get_sample_metric(self, key, metric = None):
        """ Metrics across samples as a function of time.
        """
        tmp = []
        for worm in self.samples:
            tmp.append(worm.get_data(key))
        tmp = pd.concat(tmp, axis = 1)
        if metric ==None:
            return tmp
        if metric == "mean":
            return tmp.mean(axis = 1)
        if metric == "std":
            return tmp.std(axis = 1)
        if metric == "N":
            return tmp.count(axis = 1)
        if metric == "sem":
           return tmp.std(axis = 1)/np.sqrt(len(self))
        else:
            raise Exception("Metric not implemented, choose one of 'mean', 'std', 'sem' or 'N'")


    def get_sample_metric_aligned(self, key, metric):
        """ Metrics across samples as a function of time. Uses stimulus/timepoint aligned data.
        """
        tmp = []
        for worm in self.samples:
            tmp.append(worm.get_data(key))
        tmp = pd.concat(tmp, axis = 1)
        if metric ==None:
            return tmp
        if metric == "mean":
            return tmp.mean(axis = 1)
        if metric == "std":
            return tmp.std(axis = 1)
        if metric == "N":
            return tmp.count(axis = 1)
        if metric == "sem":
           return tmp.std(axis = 1)/np.sqrt(len(self))
        else:
            raise Exception("Metric not implemented, choose one of 'mean', 'std', 'sem' or 'N'")



    ######################################
    #
    #   Plotting functions
    #
    #######################################
    def plot_2d(self, key1, key2, ax, average = True, plot_type = 'line', **kwargs):
        """plot any x-y scatter or correlation.
            Inputs:
                key1: string, column of data in the Worm object.
                key2: string, column of data in the Worm object.
                ax: either matplotlib axis object or list of axes
                average: if true, plot the sample means of each key
        """
        #each worm sample in a subplot
        if isinstance(ax, list):
            for wi, worm in enumerate(self.samples):
                if wi>len(ax):
                    warnings.warn('Too few subplots detected. Multiple samples will be plotted in a subplot.')
                ax[(wi+1)%len(ax)].plot(worm.get_data(key1), worm.get_data(key2), color = self.color, **kwargs)
        else:
            # time averaged data in a single subplot 
            if average:
                y1 = self.get_sample_metric(key1, metric = 'mean')
                y2 = self.get_sample_metric(key2, metric = 'mean')
                ax.plot(y1, y2, color = self.color, **kwargs)
            # all individual samples in one subplot
            else:
                for wi, worm in enumerate(self.samples):
                    ax.plot(worm.get_data(key1), worm.get_data(key2), **kwargs)


    def plot_kde(self, key1, key2, ax, average = True,  **kwargs):
        """plot a kernel-density estimate for two variables.
            Inputs:
                key1: string, column of data in the Worm object.
                key2: string, column of data in the Worm object.
                ax: either matplotlib axis object or list of axes
                average: if true, plot the sample means of each key
        """
        #each worm sample in a subplot
        if isinstance(ax, list):
            for wi, worm in enumerate(self.samples):
                if wi>len(ax):
                    warnings.warn('Too few subplots detected. Multiple samples will be plotted in a subplot.')
                    x, y = worm.get_data(key1), worm.get_data(key2)
                    filt = np.where(np.isfinite(x)*np.isfinite(y))
                    style.KDE_plot(ax[(wi+1)%len(ax)], x.loc[filt],y.loc[filt], color = self.color, **kwargs)
        else:
            # time averaged data in a single subplot 
            if average:
                x = self.get_sample_metric(key1, metric = 'mean')
                y = self.get_sample_metric(key2, metric = 'mean')
                filt = np.where(np.isfinite(x)*np.isfinite(y))
                style.KDE_plot(ax, x.loc[filt],y.loc[filt], color = self.color, **kwargs)
            # all individual samples in one subplot
            else:
                for wi, worm in enumerate(self.samples):
                    x, y = worm.get_data(key1), worm.get_data(key2)
                    filt = np.where(np.isfinite(x)*np.isfinite(y))
                    style.KDE_plot(ax, x.loc[filt],y.loc[filt], **kwargs)


    def plot_timeseries(self, key, ax, average = True, error = 'sem', **kwargs):
        """plot a property as a function of time.
        Inputs:
            key is a column name in the Worm object.
            ax is the matplotlib axis to plot in, if a list of axes we will plot into each separately.
            average: average all data
        """
        # each timeseries in a subplot
        if isinstance(ax, list):
            for wi, worm in enumerate(self.samples):
                if wi>len(ax):
                    raise Warning('Too few subplots detected. Multiple samples will be plotted in a subplot.')
                ax[(wi+1)%len(ax)].plot(worm.data(key), color = self.color, **kwargs)
        else:
            # average across samples
            if average:
                y = self.get_sample_metric(key, metric = 'mean')
                ax.plot(y, color = self.color, **kwargs)
                # add error band
                if error:
                    yerr = self.get_sample_metric(key, metric = error)
                    ax.fill_between(y.index, y-yerr, y+yerr, color = self.color, alpha = 0.5)
            # all individual lines but in one subplot
            else:
                for wi, worm in enumerate(self.samples):
                    ax.plot(worm.data(key), color = self.color, **kwargs)
        

    def plot_averages(self, key, ax, loc = 0, plot_style = 'box',  **kwargs):
        """plot average of a key for a sample (Worm).
        Inputs:
            key is a column name in the Worm object.
            ax is the matplotlib axis to plot in, if a list of axes we will plot into each separately.
            plot_style: one of ('box', 'bar') boxplot or barplot.
            loc: xlocation of the plot to allow adding multiple conditions in one.
            **kwargs: gets passed onto Worm.get_metric
        """
        tmpdata = []
        for worm in self.samples:
            metric = 'mean'
            # check if user gave us a different metric
            if 'metric' in kwargs:
                metric = kwargs['metric']
            tmpdata.append(worm.get_metric(key, metric))
        # plot boxplot
        if plot_style == 'box':
            style.scatterBoxplot(ax, [loc], [tmpdata], [self.color], [self.strain], **kwargs)
        elif plot_style == 'bar':
            ax.bar(loc, tmpdata, color = self.color, label = self.strain)

        
