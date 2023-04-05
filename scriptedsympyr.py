# Author: Simone Mencarelli
# Start date: 14/03/2022
# Description:
# This script simulates the SAR image retrieved from a set of scatterers placed along a range line on ground
# (flat earth approx). The aim is to verify the compression gain given by the range doppler processor and
# weighted by the antenna pattern. Effects of processing on additive noise are evaluated too.
# Version: 0
# Notes: run first the script "patterncreator.py" to have an ideal 2-d sinc-shaped antenna pattern

#%%
# IMPORTS
# classes
import os

from channel import Channel
from radar import Radar
from pointTarget import PointTarget
from dataMatrix import Data
from rangeDoppler import RangeDopplerCompressor
# py modules
import numpy as np

# switch for plotting
ifplot = True

# %%
# 0 -  SIMULATOR SETTINGS

# creating the radar object
from sar_design_equations import adjust_prf_to_looking_angle, prf_to_ground_swath

radar = Radar()
# data object
data = Data()
# channel
channel = Channel(radar)
# signal Bandwidth
bw = 2E6  # Hz (reduced for tests)
# oversampling factor
osf = 10
# sampling frequency
fs = osf * bw
# setting bandwidth and time bandwidth product in radar.pulse
radar.pulse.set_kt_from_tb(bw, 90)

# %%
# 1 - RADAR PARAMETRIZATION

# 1.1 - GEOMETRY SETTINGS
# we are assuming an antenna of size 4 x .8 m flying at an altitude:
radar_altitude = 500E3  # m
# that therefore has an average orbital speed of:
# the platform speed  # gravitational mu # earth radius
radar_speed = np.sqrt(3.9860044189e14 / (6378e3 + radar_altitude))  # m/s
# and it's looking at an off-nadir angle:
side_looking_angle_deg = 30  # deg

# setting the geometry in radar
# looking angle
radar.geometry.set_rotation(side_looking_angle_deg * np.pi / 180,
                            0,
                            0)
# initial position
radar.geometry.set_initial_position(0,
                                    0,
                                    radar_altitude)
# speed
radar.geometry.set_speed(radar_speed)

# 1.2 - PATTERN EXTENSION ON GROUND, DOPPLER PREDICTION PRF CHOICE
# to find the correct PRF we need to know thw doppler bandwidth, we assume this is the 3-dB beamwidth azimuth/doppler
# extension of the antenna pattern on ground i.e.:

# from equivalent antenna length and width:
antenna_L = 4  # m
antenna_W = .8  # m
# and operative frequency
f_c = 10E9  # Hz
# record this in radar object
radar.set_carrier_frequency(f_c)
# speed of light
c_light = 299792458  # m/s
wave_l = c_light / f_c  # m

# the approximated 0 2 0 beam-width angle given by the antenna width is:
theta_range = 2 * np.arcsin(wave_l / antenna_W)  # radians
# with a margin of nn times
theta_range *= 1
# the approximated 0 2 0 beam-width angle given by the antenna length is
theta_azimuth = 2 * np.arcsin(wave_l / antenna_L)  # radians
# with a margin of nn times
theta_azimuth *= 1

# the near range ground point is found as:
fr_g = np.tan(-radar.geometry.side_looking_angle - theta_range / 2) * radar.geometry.S_0[2]
# the far range ground point is found as:
nr_g = np.tan(-radar.geometry.side_looking_angle + theta_range / 2) * radar.geometry.S_0[2]

# the negative azimuth bea ground point is
na_g = np.tan(-theta_azimuth / 2) * radar.geometry.S_0[2] / np.cos(-radar.geometry.side_looking_angle - theta_range / 2)
# the positive azimuth ground point is
fa_g = np.tan(theta_azimuth / 2) * radar.geometry.S_0[2] / np.cos(-radar.geometry.side_looking_angle - theta_range / 2)

# the doppler extension at mid-swath is:
# from the mid-swath integration time
integration_time = np.tan(np.arcsin(wave_l / antenna_L)) * radar.geometry.S_0[2] / \
                   (np.cos(radar.geometry.side_looking_angle) * radar.geometry.abs_v)
it = - integration_time / 2
# 3-db beamwidth Doppler bandwidth:
doppler_bandwidth = float(
    2 * (-2) * radar.geometry.abs_v ** 2 * it / \
    (wave_l * np.sqrt(radar.geometry.abs_v ** 2 * it ** 2 + \
                      (radar.geometry.S_0[2] / np.cos(radar.geometry.side_looking_angle)) ** 2))
)
print(" The estimated adequate doppler bandwidth is: ", doppler_bandwidth, "Hz")

# Considering an azimuth oversampling factor of:
prf_osf = 3
# the prf will be:
radar_prf = np.abs(doppler_bandwidth) * prf_osf
radar_prf = adjust_prf_to_looking_angle(radar.geometry, radar_prf)
# setting it in:
radar_prf = data.set_prf(radar_prf, fs)
radar.set_prf(radar_prf)
print(" The PRF alligned to the sampling frequency is: ", radar_prf, "Hz")

# check to see if a pulse fits a pulse repetition intervall
if radar.pulse.duration > 1 / radar_prf:
    print("ERRORRRRRRRR impulse too long")

# %%
# 2 - TARGET PLACEMENT
tnum = 1
swath, pippo, pluto = prf_to_ground_swath(channel.radar.geometry, channel.radar.prf)
# tspacing = swath / (tnum + 0.1)
# target assigned index
target_id = 0
# target object
target = PointTarget(index=target_id)
# TEST: place the target at ground broadside
broadside_g = radar.geometry.get_broadside_on_ground()
target.set_position_gcs(broadside_g[0], broadside_g[1], 0)
print('target info')
print(target.__dict__)
# add target to the simulation
channel.add_target(target)
# for tt in range(tnum):
#     # target assigned index
#     target_id = tt
#     # target object
#     target = PointTarget(index=target_id)
#     # TEST: place the target at ground broadside
#     broadside_g = radar.geometry.get_broadside_on_ground()
#     target.set_position_gcs(broadside_g[0], broadside_g[1] + tspacing * (-int(tnum / 2) + tt), 0)
#     # target.set_position_gcs(broadside_g[0], broadside_g[1], 0)
#     # add target to the simulation
#     channel.add_target(target)

# %%
# - 3 SIMULATION and pickling
# use cubic interpolation for antenna pattern, or not
radar.antenna.cubic = False  # if the default antenna pattern is used, a linear interpolator works best
t_minmax = .5
channel.raw_signal_generator(data, -t_minmax, t_minmax, osf=osf)
# applying the compression filter to the simulated signal
channel.filter_raw_signal(data)

# %% store the simulation data for further processing
import pickle as pk

print('pickling')
filename = './Simulation_Data/channel_dump.pk'
os.makedirs(os.path.dirname(filename), exist_ok=True)
with open('./Simulation_Data/channel_dump.pk', 'wb') as handle:
    pk.dump(channel, handle)
    handle.close()
with open('./Simulation_Data/data_dump.pk', 'wb') as handle:
    pk.dump(data, handle)
    handle.close()

# %%
# - 4 RANGE DOPPLER COMPRESSION
# compress the signal imposing a finite doppler bandwidth
# create a range Doppler instance
rangedop = RangeDopplerCompressor(channel, data)
# compress the image
outimage = rangedop.azimuth_compression(doppler_bandwidth=doppler_bandwidth, patternequ=False)

# %%

if ifplot:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1)
    ax.pcolormesh(data.get_fast_time_axis(), data.get_slow_time_axis(), np.abs(outimage).astype(np.float32),
                  shading='auto', cmap=plt.get_cmap('hot'))
