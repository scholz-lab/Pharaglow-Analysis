import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
from pharaglow import io, extract

class Worm:
    """class to contain data from a single pharaglow result."""
    def __init__(self, filename, columns,fps, scale, **kwargs):
        """initialize object and load a pharaglow results file."""
        self.flag = False
        print('Reading', filename)
        # keep some metadata
        self.particle_index = int(filename.split('.')[0].split('_')[-1])
        self.experiment = filename[:6]
        # load data
        self._load(filename, columns, fps, scale)


    def _load(self, filename, columns, fps, scale, **kwargs):
        """load data."""
        traj = io.load(filename, orient='split')
        # drop all columns except the ones we want
        traj = traj.filter(columns)
        # velocity and real time
        traj['time'] = traj['frame']/fps
        #print(traj.info())
        traj['velocity'] = np.sqrt((traj['x'].diff()**2+traj['y'].diff()**2))/traj['frame'].diff()*scale*fps
        # pumping related data
        try:
            peaks, pump_clean, pks, roc, metric  = extract.bestMatchPeaks(traj['pumps'].values, **kwargs)
            # reset peaks to match frame
            peaks += np.min(traj.frame)
            traj['pump_clean'] = pump_clean
            self.pump_quality = [pks, roc, metric]
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
        """return metrics of a data column."""
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
    

    def get_data(self, key = None):
        """return a column of data or the whole pandas dataframe."""
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
        

class Experiment:
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
        """ Average across samples as a function of time.
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
    def plot_2d(self, key1, key2, ax, single=False):
        """plot all x-y trajectories."""
        
        if len(ax) == 1:
            for worm in self.samples:
                ax.plot(worm.data[key1], worm.data[key2], color = self.color)


    def plot_timeseries(self, key, ax, average = True):
        """plot a property as a function of time.
        Inputs:
            key is a column name in the Worm object.
            ax is the matplotlib axis to plot in, if a list of axes we will plot into each separately.
            average: average all data
        """
        if isinstance(ax, plt.axes):
            if average:
                for worm in self.samples:
                    ax = plt.plot(worm.get_data(key))
        

    def plot_averages(self, key, type):
        pass
    

######################################
#
#    Example code loading an experiment
#
#######################################
control = Experiment(strain='GRU101', condition='Entry', scale=2.34, fps = 30.)
control.load_data('/home/mscholz/Desktop/TestOutput_MS0006', nmax = 2)
######################################
#
#    class supports slicing
#
#######################################
# get just a few samples. a is still an Experiment object
a = control[0:2]
# if you want to access the underlying worm class, use get_sample(n) 
w = control.get_sample(0)
######################################
#
#    calculate metrics at the worm level or experiment level
#
#######################################
# metric at the worm level.
for i in ['mean', 'sem', 'std', 'N']:
    t = w.get_metric('velocity', i)
    print(t)
# metric at the experiment level.
plt.figure()
key = 'velocity'
for i, metric in enumerate(['mean', 'sem', 'std', 'N']):
    plt.subplot(2,2,i+1)
    t = a.get_sample_metric(key, metric)
    plt.plot(t)
    plt.ylabel(f"{metric} {key}")
plt.show()
######################################
#
#    calculate reversals/stimulus alignment
#
#######################################
control.calculate_reversals(animal_size=50, angle_treshold=120)
#TODO add stimulus alignment
######################################
#
#   plotting utilities
#
#######################################
# scatter two variables against each other
ax = plt.subplot(111)
control.plot_2d('velocity', 'rate', ax)