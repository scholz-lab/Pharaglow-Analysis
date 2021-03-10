from pg_analysis.plotter import Worm, Experiment
import matplotlib.pylab as plt
from pg_analysis import style
######################################
#
#    Example code loading an experiment
#
#######################################
#control = Experiment(strain='GRU101', condition='Entry', scale=2.34, fps = 30.)
#control.load_data('/home/mscholz/Desktop/TestOutput_MS0006')
#control.set_color(style.R0)
#control.dump('test')
control = Experiment.load('test')
w = control.get_sample(0)
w.multi_align([400, 700, 500], 300, 500, None)
plt.plot(w.get_aligned_metric('velocity', 'mean'))
plt.plot(w.get_aligned_metric('velocity', 'std'))
plt.plot(w.get_aligned_metric('velocity', 'N'))
plt.show()
######################################
#
#   plotting utilities
#
#######################################
# box plots!
ax = plt.subplot(111)
control.plot_averages('velocity', ax, loc = 0, plot_style = 'box')
control.plot_averages('rate', ax, loc = 1, plot_style = 'box')
# need to add our labels back when doing two plots
ax.set_xticks([0,1])
ax.set_xticklabels(['GRU101', 'This could be a second genotype'])
plt.show()
# scatter two variables against each other - density plots and normal plots
ax = plt.subplot(211)
control.plot_kde('velocity', 'rate', ax, average = True,  alpha = 0.5, markersize=10, linestyle = 'None', marker = 'o')
ax = plt.subplot(212)
control.plot_2d('x', 'y', ax, average = False,  alpha = 0.5)
plt.show()
# scatter two variables against each other with different subplot options
ax = plt.subplot(221)
ax1 = plt.subplot(222)
ax2 = plt.subplot(223)
ax3 = plt.subplot(224)
control.plot_2d('velocity', 'rate', ax, linestyle = 'None', marker = 'x')
control.plot_2d('velocity', 'rate', [ax1, ax2], alpha = 0.5)
control.plot_timeseries('velocity', ax3, average = True)
plt.show()
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
# # metric at the worm level, eg. average velocity
for i in ['mean', 'sem', 'std', 'N']:
    t = w.get_metric('velocity', i)
    print(t)


# # metric at the experiment level.
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
# # align worm data to frame 200 plus 500 minus 300 frames
plt.plot(w.align(400, 300, 500, 'velocity'))
plt.show()


