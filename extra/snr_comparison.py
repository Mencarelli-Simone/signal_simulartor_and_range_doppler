# by Simone Mencarelli
# created on 30.03.22
#
# SCOPE:this script is intended to use the results stored in memory by scripted_simulation.py and produce a
# input vs output Signal to Noise Ratio over swath range
#


#%% IMPORTS
import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from dataMatrix import Data
from channel import Channel

#%% UNPACKING RAW NOISE and compressed raw noise
# first we import the data class of interest
from rangeDoppler import RangeDopplerCompressor

with open('./Noise_Data/data_dump.pk', 'rb') as handle:
    noise_data = pk.load(handle)
    handle.close()
# load the raw input data
noise_data.load_rx_data()
# and the output image
noise_data.load_reconstructed_image()

#%% COMPUTE range bins NOISE STATISTICS
# (its independent from azimuth so we can use the azimuth lines to compute noise power)
noise_power_in = 1/noise_data.rows_num * np.sum(np.abs(noise_data.data_matrix * np.conj(noise_data.data_matrix)), axis=0)
# the input power should be 1 W approximately

# doing the same thing for the output noise
noise_power_out = 1/noise_data.rows_num * \
                  np.sum(np.abs(noise_data.reconstructed_image * np.conj(noise_data.reconstructed_image)), axis=0)

#%% plot and check
fast_time = noise_data.get_fast_time_axis()
fig, ax = plt.subplots(1)
ax.plot(fast_time, noise_power_in)
ax.set_xlabel("fast time [s]")
ax.set_ylabel("N_p in [W]")
fig, ax = plt.subplots(1)
ax.plot(fast_time, noise_power_out)
ax.set_xlabel("fast time [s]")
ax.set_ylabel("N_p out [W]")

#%% LOAD SIGNAL DATA
# first we import the data class of interest
with open('./Simulation_Data/data_dump.pk', 'rb') as handle:
    signal_data = pk.load(handle)
    handle.close()

# and the output image
signal_data.load_reconstructed_image()

peak_index = np.argmin(np.abs(signal_data.get_slow_time_axis() - 0))
# find the output peak line
signal = signal_data.reconstructed_image[peak_index, :]
signal_up = signal_data.reconstructed_image[peak_index + 1, :]
signal_down = signal_data.reconstructed_image[peak_index - 1, :]

#%% plot and check
fig, ax = plt.subplots(1)
ax.plot(fast_time, np.abs(signal))
ax.plot(fast_time, np.abs(signal_up), label='up')
ax.plot(fast_time, np.abs(signal_down), label='dwn')
ax.set_xlabel("fast time [s]")
ax.set_ylabel("signal at azimuth 0")
ax.legend()


#%% compute SNR
SNR_o = np.abs(signal*np.conj(signal)) / noise_power_out

fig, ax = plt.subplots(1)
ax.plot(fast_time, 10* np.log10(SNR_o))
ax.set_xlabel("fast time [s]")
ax.set_ylabel("SNRo")

# note the SNR_input is one in any case, so we can just use SNR_o from now on

#%% SNR calibration i.e. adding the system parameters to complete the radar equation
# we need the radar parameters here
with open('./Simulation_Data/channel_dump.pk', 'rb') as handle:
    channel = pk.load(handle)
    handle.close()
# we also need some methods from rangedoppler
rdop = RangeDopplerCompressor(channel, signal_data)

sigma = 1 # target reflectivity
noise_f = 10**(5 / 10) # noise figure
# average transmitted power
powav = 15 # W
# p peak
ppeak = powav / (channel.radar.pulse.duration * channel.radar.prf)
# antenna noise temperature
T_ant = 300 # k
# wavelength
wave_l = channel.c / channel.radar.fc
# beam center side looking range from ground
average_range = rdop.r_0 # this is actually the broadside range,
# the midswath range shall be computed using actual beamwidth information
# note: the signal generator doesn't incorporate free space loss, this is here estimated

# doppler bandwidth standard calculation, if bigger than PRF, PRF shall be chosen
antenna_L = channel.radar.antenna.L
integration_time = np.tan(np.arcsin(wave_l / antenna_L)) * channel.radar.geometry.S_0[2] / \
                   (np.cos(channel.radar.geometry.side_looking_angle) * channel.radar.geometry.abs_v)
it = - integration_time / 2
# 3-db beamwidth Doppler bandwidth:
doppler_bandwidth = float(
                    2 * (-2) * channel.radar.geometry.abs_v**2 * it / \
                    (wave_l * np.sqrt(channel.radar.geometry.abs_v**2 * it**2 + \
                                      (channel.radar.geometry.S_0[2] / np.cos(channel.radar.geometry.side_looking_angle))**2))
                    )

# azimuth resolution, given by doppler bandwidth
dx = channel.radar.geometry.abs_v / doppler_bandwidth
# ground range resolution, given by range bandwidth and looking angle
sin_eta = np.sqrt(rdop.get_true_range_axis()**2 - channel.radar.geometry.S_0[2,0]**2) / rdop.get_true_range_axis()
dy = channel.c / (2 * channel.radar.pulse.get_bandwidth() * sin_eta)
k_boltz = 1.380649E-23 # J/K
# using true range axis for the range
scaling = ppeak * wave_l ** 2 * sigma * dx * dy / \
          ((4* np.pi) **3 * rdop.get_true_range_axis()**4 * noise_f * k_boltz * T_ant * channel.radar.pulse.get_bandwidth())

SNR_final = scaling * SNR_o # todo check, this is wrong. the input signal is not normalized or it is
SNR_noise = scaling * channel.radar.antenna.broadside_gain**2 * (channel.radar.pulse.get_bandwidth()*doppler_bandwidth)**2 / noise_power_out
#%% plot
# convert range to ground range
ground_range_axis = -rdop.get_true_range_axis() * sin_eta

fig, ax = plt.subplots(1)
ax.plot(ground_range_axis, 10 * np.log10(SNR_final))
ax.set_xlabel("ground range [m]")
ax.set_ylabel("SNR")
#%%
fig, ax = plt.subplots(1)
ax.plot(ground_range_axis, 10 * np.log10(SNR_noise))
ax.set_xlabel("ground range [m]")
ax.set_ylabel("SNR")


#%% pickle to compare with theoretical
import os
path = './SNR_simulation'
if not os.path.exists(path):
    os.makedirs(path)

name_snr = '/snr.pk'
filename = path + name_snr
with open(filename, 'wb') as handle:
    pk.dump(SNR_final, handle)
    handle.close()

name_snr = '/snr_noise.pk'
filename = path + name_snr
with open(filename, 'wb') as handle:
    pk.dump(SNR_noise, handle)
    handle.close()

name_ax = '/gnd_rng.pk'
filename = path + name_ax
with open(filename, 'wb') as handle:
    pk.dump(ground_range_axis, handle)
    handle.close()




