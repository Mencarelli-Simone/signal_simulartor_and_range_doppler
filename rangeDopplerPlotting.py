from numba import jit

from dataMatrix import Data
import pickle as pk
import matplotlib.pyplot as plt
import numpy as np
from channel import Channel

# %% unpickle the data
from rangeDoppler import r_of_f_r_dopl

with open('./Simulation_Data/channel_dump.pk', 'rb') as handle:
    channel = pk.load(handle)
    handle.close()
with open('./Simulation_Data/data_dump.pk', 'rb') as handle:
    data = pk.load(handle)
    handle.close()
# load everything from file
data.load_all()

# %% axes
range_ax = data.get_range_axis(channel.c)
fast_time_ax = data.get_fast_time_axis()

azimuth_ax = data.get_azimuth_axis(channel.radar.geometry.abs_v)
slow_time_ax = data.get_slow_time_axis()


# conversion
def t_to_r(t):
    return t * channel.c / 2


def r_to_t(r):
    return 2 * r / channel.c


def s_to_a(s):
    return s * channel.radar.geometry.abs_v


def a_to_s(a):
    return a / channel.radar.geometry.abs_v


doppler_ax = data.get_doppler_axis(prf=channel.radar.prf, doppler_centroid=0)

# %% range doppler plot axes and curves

# expected return loci
rc = np.sqrt(500e3 ** 2 + channel.target[0].pos_gcs[1] ** 2)
rd_dop = r_of_f_r_dopl(doppler_ax, rc, channel.radar.geometry.abs_v, channel.c / channel.radar.fc, channel.c,
                       channel.radar.pulse.rate)
td_dop = (2 * rd_dop / channel.c) % (1 / channel.radar.prf)

# %% time domain plot
fig, (ax, cax) = plt.subplots(1, 2,
                              gridspec_kw={"width_ratios": [1, 0.04]},
                              figsize=[8 * 1.07, 4.8])

fig.canvas.manager.set_window_title('R-A range compressed')
c = ax.pcolormesh(fast_time_ax, slow_time_ax, np.abs(data.data_range_matrix),
                  shading='auto')
ax.set_xlabel("fast time [s]")
ax.set_ylabel("slow time [s]")
# # ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
secaxy = ax.secondary_yaxis('right', functions=(s_to_a, a_to_s))
secaxy.set_ylabel("azimuth [m]")
fig.colorbar(c, cax=cax)
fig.subplots_adjust(wspace=0.33)
fig.tight_layout()

# %% raw
# plot
fig, (ax, cax) = plt.subplots(1, 2,
                              gridspec_kw={"width_ratios": [1, 0.04]},
                              figsize=[6.2, 4])  # [8,4.8])
fig.canvas.manager.set_window_title('R-A range uncompressed')
c = ax.pcolormesh(fast_time_ax, slow_time_ax, np.real(data.data_matrix),
                  shading='auto', rasterized=True)
ax.set_xlabel("fast time [s]")
ax.set_ylabel("slow time [s]")
# # ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
secaxy = ax.secondary_yaxis('right', functions=(s_to_a, a_to_s))
secaxy.set_ylabel("azimuth [m]")
fig.colorbar(c, cax=cax)
fig.subplots_adjust(wspace=0.33)
fig.tight_layout()
# %% raw cuts
fig, ax = plt.subplots(1, 1, figsize=[7.5, 7])  # [8,4.8])
fig.canvas.manager.set_window_title('R-A range uncompressed h cut')
ax.plot(fast_time_ax, np.real(data.data_matrix[int(data.rows_num / 2 + 0.5), :]))
ax.set_xlabel("fast time [s]")
# ax.set_ylabel("slow time [s]")
# # ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
# secaxy = ax.secondary_yaxis('right', functions=(s_to_a, a_to_s))
# secaxy.set_ylabel("azimuth [m]")

fig.subplots_adjust(wspace=0.33)
fig.tight_layout()
# %%
fig, ax = plt.subplots(1, 1, figsize=[7.5, 7])  # [8,4.8])
fig.canvas.manager.set_window_title('R-A range uncompressed v cut')
# ax.plot(np.real(data.data_matrix[:, int((rc % range_ax[-1]) / (range_ax[1] - range_ax[0]))]),slow_time_ax, )

index = int(np.round(((rc * 2 / channel.c) % (1 / data.prf)) * data.Fs))
ax.plot(np.real(data.data_matrix[:, index]), slow_time_ax)
# ax.set_xlabel("fast time [s]")
ax.set_ylabel("slow time [s]")
# # ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
# secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
# secax.set_xlabel("enveloped range [m]")
secaxy = ax.secondary_yaxis('right', functions=(s_to_a, a_to_s))
secaxy.set_ylabel("azimuth [m]")

fig.subplots_adjust(wspace=0.33)
fig.tight_layout()

# %% plot
fig, (ax, cax) = plt.subplots(1, 2,
                              gridspec_kw={"width_ratios": [1, 0.04]},
                              figsize=[8, 4.8])
fig.canvas.manager.set_window_title('R-D range compressed')
c = ax.pcolormesh(fast_time_ax, doppler_ax, np.abs(data.doppler_range_compressed_matrix),
                  shading='auto')
ax.plot(td_dop, doppler_ax, '--r')  # with doppler error included

ax.set_xlabel("fast time [s]")
ax.set_ylabel("Doppler freq. [Hz]")
# ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
fig.colorbar(c, cax=cax)
fig.subplots_adjust(wspace=0.33)
fig.tight_layout()

# %% rcmc plot
fig, (ax, cax) = plt.subplots(1, 2,
                              gridspec_kw={"width_ratios": [1, 0.04]},
                              figsize=[8, 7])
fig.canvas.manager.set_window_title('R-D rcmc')
c = ax.pcolormesh(fast_time_ax, doppler_ax, np.abs(data.doppler_range_compressed_matrix_rcmc),
                  shading='auto')
ax.plot(td_dop, doppler_ax, '--r')  # with doppler error included

ax.set_xlabel("fast time [s]")
ax.set_ylabel("Doppler freq. [Hz]")
# ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
fig.colorbar(c, cax=cax)
fig.subplots_adjust(wspace=0.33)
fig.tight_layout()

# %% filtered plot doppler

fig, (ax, cax) = plt.subplots(1, 2,
                              gridspec_kw={"width_ratios": [1, 0.04]},
                              figsize=[8, 4.8])
fig.canvas.manager.set_window_title('R-D reconstructed')
c = ax.pcolormesh(fast_time_ax, doppler_ax, np.abs(data.range_doppler_reconstructed_image),
                  shading='auto')
# ax.plot(td_dop, doppler_ax, '--r')  # with doppler error included

ax.set_xlabel("fast time [s]")
ax.set_ylabel("Doppler freq. [Hz]")
# ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
fig.colorbar(c, cax=cax)
fig.subplots_adjust(wspace=0.33)
fig.tight_layout()

# %% filtered plot time
fig, (ax, cax) = plt.subplots(1, 2,
                              gridspec_kw={"width_ratios": [1, 0.04]},
                              figsize=[6.2, 4])
fig.canvas.manager.set_window_title('R-A reconstructed')
c = ax.pcolormesh(fast_time_ax, slow_time_ax, np.abs(data.reconstructed_image),
                  shading='auto', cmap=plt.get_cmap('hot'), rasterized=True)
ax.set_xlabel("fast time [s]")
ax.set_ylabel("slow time [s]")
# ax.set_xlim(9.87e-05 - .387e-5,9.87e-05 + .387e-5)
## ax.set_ylim(0.2 - 5e-3,.2 + 5e-3)
secax = ax.secondary_xaxis('top', functions=(t_to_r, r_to_t))
secax.set_xlabel("enveloped range [m]")
secaxy = ax.secondary_yaxis('right', functions=(s_to_a, a_to_s))
secaxy.set_ylabel("azimuth [m]")
fig.colorbar(c, cax=cax)
fig.subplots_adjust(wspace=0.33)
fig.tight_layout()
# %%
xmagnification = 50
ymagnification = 700
ax.set_xlim(fast_time_ax[-1] * (1 / 2 - 1 / xmagnification ) - .5e-6, fast_time_ax[-1] * (1 / 2 + 1 / xmagnification))
ax.set_ylim((slow_time_ax[-1] - slow_time_ax[0]) * (- 1 / ymagnification),
            (slow_time_ax[-1] - slow_time_ax[0]) * (+ 1 / ymagnification))
fig.tight_layout()
plt.show()

# %%

central_line = data.reconstructed_image[int(data.rows_num / 2), :]
central_line_up = data.reconstructed_image[int(data.rows_num / 2) + 1, :]
central_line_dwn = data.reconstructed_image[int(data.rows_num / 2) - 1, :]
fig, ax = plt.subplots(1)
ax.plot(fast_time_ax, np.abs(central_line))
ax.plot(fast_time_ax, np.abs(central_line_up), label='+1')
ax.plot(fast_time_ax, np.abs(central_line_dwn), label='-1')
ax.legend()
