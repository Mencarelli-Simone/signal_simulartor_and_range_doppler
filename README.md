This project contains some code to simulate a point scatterer radar signal as received by a moving platform at a constant altitude in a flat-earth geometry.\

The file "scriptedsympyr.py" contains an example of how to set up a simulation for a single scatterer, any arbitrary number of scatterers can be placed in the scene using the "channel.add_target()" method. This will simulate all the targets independently and sum together the resulting signals (i.e. more targets longer simulation times). The signal is then compressed using the range doppler algorithm.\

The bandwidth and simulation time are kept small in this example on purpose. The simulator will generate, in fact a complete received signal with a potentially large memory occupancy for large bandwidth or long signals. In fact, during the simulation and the compression multiple copies of the signal might be present in memory.\

The first time "scriptedsympyr.py" is run, it will create 3 folders: 
1. Antenna_Pattern: this contains a pickled (serialized) "Pattern" class containing the antenna directivity pattern for the radar, by default a rectangulat "boxcar" antenna pattern to produce an ideal return with finite doppler bandwidth. Any antenna pattern can be fed to the simulator by changing the gain_pattern.pk file in this folder.
2. Target_Data: this contains a number of serialized target objects 
3. Simulation_Data: this contains several files containing serialized (using the python module pickle) instances of classes used during the simulation or the range-doppler compression.  The data matrices produced aftetr each step of the comoression algorithm are stored as independent files. The Data object contained in the data_dump.pk file contains the methods to properly load these matrices from file e.g. to be plotted.

The file "rangeDopplerPlotting.py" extracts the simulation data and produces a plot corresponding to every step of the Range-Doppler algorithm.


-- Apr. 2023, Simone Mencarelli--
