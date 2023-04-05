# Author: Simone Mencarelli
# Start date: 14/03/2022
# Description:
# This script simulates the SAR image retrieved from a set of scatterers placed along a range line on ground
# (flat earth approx). The aim is to verify the compression gain given by the range doppler processor and
# weighted by the antenna pattern. Effects of processing on additive noise are evaluated too.
# Version: 0
# Notes: run first the script "patterncreator.py" to have an ideal 2-d sinc-shaped antenna pattern

# %%
# IMPORTS
# classes
import os

from channel import Channel
from interpolator_v2 import sphere_interp
from radar import Radar
from pointTarget import PointTarget
from dataMatrix import Data
from rangeDoppler import RangeDopplerCompressor
# py modules
import numpy as np

# %%
# 0 -  SIMULATOR SETTINGS

# creating the radar object
from utils import mesh_doppler_to_azimuth, mesh_azimuth_range_to_ground_gcs, mesh_gcs_to_lcs, meshCart2sph

radar = Radar()
# data object
data = Data()
# channel
channel = Channel(radar)
# signal Bandwidth
bw = 5E6 # Hz (reduced for tests)
# oversampling factor
osf = 10
# sampling frequency
fs = osf * bw
# setting this in radar
radar.pulse.set_kt_from_tb(bw, 80)


# %%
# 1 - RADAR PARAMETRIZATION

# 1.1 - GEOMETRY SETTINGS
# we are assuming an antenna of size 4 x .8 m flying at an altitude:
radar_altitude = 500E3  # m
# that therefore has an average orbital speed of:
# the platform speed  # gravitational mu # earth radius
radar_speed = np.sqrt(3.9860044189e14 / (6378e3 + radar_altitude))  # m/s
# and it's looking at an off-nadir angle:
side_looking_angle_deg = 40  # deg

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
c_light = 299792458 # m/s
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
                    2 * (-2) * radar.geometry.abs_v**2 * it / \
                    (wave_l * np.sqrt(radar.geometry.abs_v**2 * it**2 + \
                                      (radar.geometry.S_0[2] / np.cos(radar.geometry.side_looking_angle))**2))
                    )
print(" The estimated adequate doppler bandwidth is: ", doppler_bandwidth,"Hz")

# Considering an azimuth oversampling factor of:
prf_osf = 7
# the prf will be:
radar_prf = np.abs(doppler_bandwidth) * prf_osf
# setting it in:
radar_prf = data.set_prf(radar_prf, fs)
radar.set_prf(radar_prf)
print(" The PRF alligned to the sampling frequency is: ", radar_prf, "Hz")

# check to see if a pulse fits a pulse repetition intervall
if radar.pulse.duration > 1 / radar_prf:
    print("ERRORRRRRRRR impulse too long")

# %%
# 2 - TARGET PLACEMENT ( this section should be parallelized for N targets)

# target assigned index
target_id = 0
# target object
target = PointTarget(index=target_id)
# TEST: place the target at ground broadside
# todo: review and extend to multiple targets maybe this should be done inside channel
broadside_g = radar.geometry.get_broadside_on_ground()
target.set_position_gcs(broadside_g[0], broadside_g[1], 0)
# add target to the simulation
channel.add_target(target)
# add a target 100 m to a side, just to test multithreading
#t1 = PointTarget(index=target_id+1)
# t1.set_position_gcs(broadside_g[0], broadside_g[1] + 100, 0)
# channel.add_target(t1)

# %%
# - 3 SIMULATION and pickling
# use cubic interpolation for antenna pattern
radar.antenna.cubic = False
channel.raw_signal_generator_multithread(data,-.3, .3, osf=osf)
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
# - 4 NOISE SIGNAL GENERATION
# todo

# %%
# - 5 RANGE DOPPLER COMPRESSION
# compress the signal imposing a finite doppler bandwidth
# create a range Doppler instance
rangedop = RangeDopplerCompressor(channel, data)
# compress the image
outimage = rangedop.azimuth_compression(doppler_bandwidth=doppler_bandwidth)

# %%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
ax.pcolormesh(data.get_fast_time_axis(),data.get_slow_time_axis(),np.abs(outimage).astype(np.float32),shading='auto',cmap=plt.get_cmap('hot'))

# %%
# - 5 ANTENNA PATTERN EQUALIZATION
# create the range-doppler equalization filter hint use scipy interpolate no
# load range doppler rcmc-ed filtered data
# get range and doppler axes
# generate range-doppler scene weighting pattern
# a different weight for each range cell
# divide data by this equalization filter
# perform ifft, update compressed matrix

# note : rangedop.doppler_axis is available only after the creation of the doppler compression filter
rd_pattern = rangedop.create_azimuth_equalization_matrix(data.get_range_axis(), rangedop.doppler_axis)

#%%
fig, ax = plt.subplots(1)
ax.pcolormesh(data.get_range_axis(),rangedop.doppler_axis,np.abs(rd_pattern).astype(np.float32),shading='auto',cmap=plt.get_cmap('hot'))

#%% equalization matrix sequence test, exploding all the steps inside create_azimuth_equalization_matrix
range_axis, doppler_axis = data.get_range_axis(), rangedop.doppler_axis
R, D = np.meshgrid(range_axis, doppler_axis)
# R as it is is the enveloped range, we need the true range so:
midswath_range = rangedop.radar.geometry.get_broadside_on_ground()
midswath_range = rangedop.radar.geometry.get_lcs_of_point(midswath_range, np.zeros(1))
midswath_range = np.linalg.norm(midswath_range)
pri_offset = int(rangedop.radar.prf * midswath_range * 2 / rangedop.c)
range_offset = pri_offset * rangedop.c / (2 * rangedop.radar.prf)
R += range_offset
#%%
# transform to range azimuth points
# using posp time-frequency locking relation for slow time:
R, A = mesh_doppler_to_azimuth(R, D, rangedop.c / rangedop.radar.fc, rangedop.radar.geometry.abs_v)
#%%
# transform to ground gcs points
X, Y = mesh_azimuth_range_to_ground_gcs(R, A, rangedop.radar.geometry.velocity, rangedop.radar.geometry.S_0)
#%%
# tronsform to lcs points
X, Y, Z = mesh_gcs_to_lcs(X, Y, np.zeros_like(X), rangedop.radar.geometry.Bc2s, rangedop.radar.geometry.S_0)
#%%
# transform to spherical
R, Th, Ph = meshCart2sph(X, Y, Z)
Ph = np.where(Ph < 0, np.pi * 2 + Ph, Ph)
#%%
fig, ax = plt.subplots(1)
c = ax.pcolormesh(Th * np.cos(Ph), Th * np.sin(Ph), np.ones_like(Th))
# c = ax.pcolormesh(Phi * cos(Theta), Phi * sin(Theta) ,10*np.log10(np.abs(pat.gain_pattern)), cmap=plt.get_cmap('hot'))
fig.colorbar(c)
ax.set_xlabel("$\\theta\  cos \phi$")
ax.set_ylabel("$\\theta\  sin \phi$")
ax.set_xlim(-np.pi/32, np.pi/32)
ax.set_ylim(-np.pi/32, np.pi/32)

#%%
# find pattern over grid need a spherical interpolator method for meshgrids
r_d_gain = 1j * np.zeros_like(np.ravel(Th))
r_d_gain = sphere_interp(np.ravel(Th.T),
                         np.ravel(Ph.T),
                         radar.antenna.theta_ax,
                         radar.antenna.phi_ax,
                         radar.antenna.gain_matrix,
                         r_d_gain,
                         cubic=False)
r_d_gain = r_d_gain.reshape(Th.T.shape)

#%%
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1)
c = ax.pcolormesh(Th * np.cos(Ph), Th * np.sin(Ph), 10*np.log10(np.abs(r_d_gain.T)) , cmap=plt.get_cmap('hot'))
# c = ax.pcolormesh(Phi * cos(Theta), Phi * sin(Theta) ,10*np.log10(np.abs(pat.gain_pattern)), cmap=plt.get_cmap('hot'))
fig.colorbar(c)
ax.set_xlabel("$\\theta\  cos \phi$")
ax.set_ylabel("$\\theta\  sin \phi$")
ax.set_xlim(-np.pi/32, np.pi/32)
ax.set_ylim(-np.pi/32, np.pi/32)

#%%
import matplotlib.pyplot as plt
tt, pp = np.meshgrid(radar.antenna.theta_ax, radar.antenna.phi_ax)
fig, ax = plt.subplots(1)
c = ax.pcolormesh(tt * np.cos(pp), tt * np.sin(pp), 10*np.log10(np.abs(radar.antenna.gain_matrix.T)) , cmap=plt.get_cmap('hot'))
# c = ax.pcolormesh(Phi * cos(Theta), Phi * sin(Theta) ,10*np.log10(np.abs(pat.gain_pattern)), cmap=plt.get_cmap('hot'))
fig.colorbar(c)
ax.set_xlabel("$\\theta\  cos \phi$")
ax.set_ylabel("$\\theta\  sin \phi$")
ax.set_xlim(-np.pi/32, np.pi/32)
ax.set_ylim(-np.pi/32, np.pi/32)