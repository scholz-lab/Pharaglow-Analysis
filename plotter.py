import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
from pharaglow import io, extract

class Worm:
    def __init__(self, filename, columns,fps, scale, **kwargs):
        """load a pharaglow results file."""
        self.flag = False
        print('Reading', filename)
        # keep some metadata
        self.particle_index = int(filename.split('.')[0].split('_')[-1])
        self.experiment = filename[:6]
        # load data
        self._load(filename, columns, fps, scale)


    def _load(self, filename, columns, fps, scale, **kwargs):
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
        if key == None:
            return self.data
        else:
            assert key in self.data.columns, f'The key {key} does not exist in the data.'
            return self.data[key]
    
    def calculate_reversals(self, angle_treshold):
        pass



class Experiment:
    """Wrapper class which is a container for individual worms."""
    # class attributes
    def __init__(self, strain, condition, scale, fps, samples = None):
        self.strain = strain
        self.condition = condition
        if samples == None:
            self.samples = []
        else:
            self.samples = samples[:]
        self.scale = scale
        self.fps = fps

    
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
    
    ######################################
    #
    #   get/set attributes
    #
    #######################################
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
    def plot_trajectories(self, ax):
        """plot all x-y trajectories."""
        for worm in self.samples:
            ax.plot(worm.data.x, worm.data.y)

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
    

""" Example code loading an experiment."""
control = Experiment(strain='GRU101', condition='Entry', scale=2.34, fps = 30.)
control.load_data('/home/mscholz/Desktop/TestOutput_MS0006', nmax = 2)
# get one worm
a = control[0:2]
print(a.get_sample_metric('velocity', 'N'))
for i, metric in enumerate(['mean', 'sem', 'std', 'N']):
    plt.subplot(2,2,i+1)
    t = a.get_sample_metric('velocity', metric)
    plt.plot(t)
plt.show()

plt.show()
a.plot_trajectories(plt.gca())
plt.show()
w = control.get_sample(0)
for i in ['mean', 'sem', 'std', 'N']:
    t = w.get_metric('velocity', i)
    print(t)
plt.show()