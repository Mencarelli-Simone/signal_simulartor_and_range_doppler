from numba import jit

from dataMatrix import Data
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np

#%% unpickle the data
from rangeDoppler import r_of_f_r_dopl

with open('./channel_dump_ideal.pk', 'rb') as handle:
    channel_id = pk.load(handle)
    handle.close()
with open('./data_dump_ideal.pk', 'rb') as handle:
    data_id = pk.load(handle)
    handle.close()
    
with open('channel_dump.pk', 'rb') as handle:
    channel_in = pk.load(handle)
    handle.close()
with open('data_dump.pk', 'rb') as handle:
    data_in = pk.load(handle)
    handle.close()
    
#%% axes
range_ax = data_id.get_range_axis(channel_id.c)
fast_time_ax = data_id.get_fast_time_axis()

azimuth_ax = data_id.get_azimuth_axis(channel_id.radar.geometry.abs_v)
slow_time_ax = data_id.get_slow_time_axis()

# conversion
def t_to_r(t):
    return t * channel_id.c / 2
def r_to_t(r):
    return 2 * r / channel_id.c

def s_to_a(s):
    return s * channel_id.radar.geometry.abs_v
def a_to_s(a):
    return a / channel_id.radar.geometry.abs_v


doppler_ax = data_id.get_doppler_axis(prf = channel_id.radar.prf, doppler_centroid = 0)

#%% raw
# plot
fig, (ax,cax) = plt.subplots(1,2,
                            gridspec_kw={"width_ratios":[1,0.04]},
                             figsize = [8,4.8])
fig.canvas.manager.set_window_title('R-A range uncompressed ideal')
c = ax.pcolormesh(fast_time_ax, doppler_ax, np.real(data_id.data_matrix),
                  shading='auto')

ax.set_xlabel("fast time [s]")
ax.set_ylabel("Doppler freq. [Hz]")
#ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
fig.colorbar(c, cax=cax)
fig.subplots_adjust(wspace = 0.33)
fig.tight_layout()

#%% raw
# plot
fig, (ax,cax) = plt.subplots(1,2,
                            gridspec_kw={"width_ratios":[1,0.04]},
                             figsize = [8,4.8])
fig.canvas.manager.set_window_title('R-A range uncompressed interpolated')
c = ax.pcolormesh(fast_time_ax, doppler_ax, np.real(data_in.data_matrix),
                  shading='auto')

ax.set_xlabel("fast time [s]")
ax.set_ylabel("Doppler freq. [Hz]")
#ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
fig.colorbar(c, cax=cax)
fig.subplots_adjust(wspace = 0.33)
fig.tight_layout()
    
#%% filtered plot time ideal
fig, (ax,cax) = plt.subplots(1,2,
                            gridspec_kw={"width_ratios":[1,0.04]},
                             figsize = [8*1.07,4.8])
fig.canvas.manager.set_window_title('R-A reconstructed ideal pattern')
c = ax.pcolormesh(fast_time_ax, slow_time_ax, np.abs(data_id.reconstructed_image),
                  shading='auto')
ax.set_xlabel("fast time [s]")
ax.set_ylabel("slow time [s]")
ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
ax.set_ylim(0.2 - 5e-3,.2 + 5e-3)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
secaxy = ax.secondary_yaxis('right', functions=(s_to_a, a_to_s))
secaxy.set_ylabel("azimuth [m]")
fig.colorbar(c, cax=cax)
fig.subplots_adjust(wspace = 0.33)
fig.tight_layout()

#%% filtered plot time interpolated
fig, (ax,cax) = plt.subplots(1,2,
                            gridspec_kw={"width_ratios":[1,0.04]},
                             figsize = [8*1.07,4.8])
fig.canvas.manager.set_window_title('R-A reconstructed interp pattern')
c = ax.pcolormesh(fast_time_ax, slow_time_ax, np.abs(data_in.reconstructed_image),
                  shading='auto')
ax.set_xlabel("fast time [s]")
ax.set_ylabel("slow time [s]")
ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
ax.set_ylim(0.2 - 5e-3,.2 + 5e-3)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
secaxy = ax.secondary_yaxis('right', functions=(s_to_a, a_to_s))
secaxy.set_ylabel("azimuth [m]")
fig.colorbar(c, cax=cax)
fig.subplots_adjust(wspace = 0.33)
fig.tight_layout()

#%% comparison

fig, (ax,cax) = plt.subplots(1,2,
                            gridspec_kw={"width_ratios":[1,0.04]},
                             figsize = [8*1.07,4.8])
fig.canvas.manager.set_window_title('R-A reconstructed compared')
c = ax.pcolormesh(fast_time_ax, slow_time_ax, np.abs(data_in.reconstructed_image-data_id.reconstructed_image),
                  shading='auto', cmap=plt.get_cmap('jet'), alpha=1, vmax=np.abs(data_in.reconstructed_image).max())
ax.set_xlabel("fast time [s]")
ax.set_ylabel("slow time [s]")
ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
ax.set_ylim(0.2 - 5e-3,.2 + 5e-3)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
secaxy = ax.secondary_yaxis('right', functions=(s_to_a, a_to_s))
secaxy.set_ylabel("azimuth [m]")
fig.colorbar(c, cax=cax)
fig.subplots_adjust(wspace = 0.33)
fig.tight_layout()

#%% cut plots
# target pos parameters
speed = channel_id.radar.geometry.abs_v
# target position
rc = np.sqrt(408e3 ** 2 + 15e3 ** 2)
t_0 = 2 * rc / channel_id.c
az_0 = 1531
s_0 = az_0 / speed
t0_index = t_0 % (1 / channel_id.radar.prf)
print(t0_index)
t0_index = np.round(t0_index * data_id.Fs).astype('int')
s0_index = np.round((s_0-slow_time_ax[0]) * channel_id.radar.prf).astype('int')

fig, ax = plt.subplots(2,1)
ax[0].plot(data_id.get_range_axis(channel_id.c),np.abs(data_in.reconstructed_image[s0_index,:]),label='interpolated')
ax[0].plot(data_id.get_range_axis(channel_id.c),np.abs(data_id.reconstructed_image[s0_index,:]),label='ideal')
ax[0].set_xlim((channel_id.c/2)*(-.7e-5+t_0 % (1/channel_id.radar.prf)), (channel_id.c/2)*(.7e-5+t_0 % (1/channel_id.radar.prf)))
ax[0].legend(loc='upper right')
ax[0].set_xlabel('enveloped range [m]')
ax[1].plot(data_id.get_azimuth_axis(7.66e3), np.abs(data_in.reconstructed_image[:, t0_index]),label='interpolated')
ax[1].plot(data_id.get_azimuth_axis(7.66e3), np.abs(data_id.reconstructed_image[:, t0_index]),label='ideal.')
ax[1].set_xlim((-0.004+s_0)*7.66e3,7.66e3*(0.004+s_0))
ax[1].set_xlabel('azimuth [m]')
ax[1].legend(loc='upper right')
ax[0].set_title('pulse orthogonal main cuts')
ax[0].grid()
ax[1].grid()
