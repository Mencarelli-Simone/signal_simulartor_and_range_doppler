import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from numpy import fft
from tqdm import tqdm
import scipy.special as sc
import pickle as pk
from radar import Radar
from channel import Channel
#%%%
# doppler in time
def fd_r(t, rc, v, lam_c):
    r1 = v ** 2 * t / (np.sqrt(rc ** 2 + t ** 2 * v ** 2))
    f = - 2 * r1 / lam_c
    return f

# fresnel
#@jit(nopython=False)
def modified_fresnel(gamma):
    S, C = sc.fresnel(gamma * np.sqrt(2 / np.pi))
    return C + S * 1j

# linearfmspectrum
#@jit(nopython=True, parallel=True)
def linearfmspectrum(f, k, tau):
    rate = k
    duration = tau
    alpha = np.abs(np.pi * rate) # alpha needs to be taken as absolute
    omega = np.pi * 2 * f
    S = np.sqrt(np.pi / (2 * np.abs(alpha))) * np.exp(-1j * omega ** 2 / (4 * alpha)) * (
            modified_fresnel((omega + np.abs(alpha) * duration) / (2 * np.sqrt(np.abs(alpha)))) - modified_fresnel(
        (omega - np.abs(alpha) * duration) / (2 * np.sqrt(np.abs(alpha)))))
    # the spectrum is calculated assuming an up chirp, if is a down chirp we apply the time reversal property of the
    # fourier transform
    if k < 0:
        S = S.conjugate()
    return S


# pulseresponsetheory
#@jit(nopython=True)
def imagepulse(s, t, t0, s0, lamb, b_r, b_d, speed, c):
    """
    :param s: slow time
    :param t: fast time (true)
    :param t0: fast encounter time
    :param s0: slow encounter time
    :param lamb: wavelength
    :param b_r: impulse bandwidth
    :param b_d: doppler bandwidth
    :param speed: satellite speed
    :param c: light speed
    :return: pixel
    """
    rc = c * t0 / 2
    kd = lamb * c * (t0 - t) / (4 * speed ** 2)
    out = np.sin(np.pi * b_r * (t - t0))/(np.pi * b_r * (t - t0)) * np.exp(-1j * 4 * np.pi * rc / lamb) * \
          np.exp(1j * np.pi / 4) * linearfmspectrum(-(s-s0), kd, Bd)
    #out = np.sin(np.pi * b_r * (t - t0))/(np.pi * b_r * (t - t0)) * np.exp(-1j * 4 * np.pi * rc / lamb) * b_d * np.sinc(np.pi * b_d * (s - s0))
    return out


#@jit(nopython=True, parallel=True)
def impulse_image(matrix, s0, t0, lamb, Br, Bd, speed, c, t_time_ax, s_time_ax):
    # for every range (fast time) cell
    for rr in tqdm(range(len(t_time_ax))):
        # for every azimuth (slow time) bin
        for aa in range(len(s_time_ax)):
            matrix[aa, rr] = imagepulse(s_time_ax[aa], t_time_ax[rr], t0, s0, lamb, Br, Bd, speed, c)
    return matrix


# %% CODE BEGIN
# import object for the fast and slow time
print('unpickling')
with open('./Simulation_Data/channel_dump.pk', 'rb') as handle:
    channel = pk.load(handle)
with open('./Simulation_Data/data_dump.pk', 'rb') as handle:
    data = pk.load(handle)

data.load_all()

fast_time_ax = data.get_fast_time_axis()
slow_time_ax = data.get_slow_time_axis()

# side looking angle
beta = channel.radar.geometry.side_looking_angle
# platform height
height = channel.radar.geometry.S_0[2]
r_0 = (height / np.cos(beta))

# minimum delay in the visible image
min_delay = np.floor((r_0 * 2 / channel.c) * channel.radar.prf) / channel.radar.prf
# true fast time
true_fast_time_ax = fast_time_ax + min_delay

# %% Create matrix
image = 1j*np.zeros((len(slow_time_ax), len(fast_time_ax)))

# parameters
# radar speed
speed = channel.radar.geometry.abs_v
# target position
rc = np.sqrt(500e3 ** 2 + 15e3 ** 2)
t_0 = 2 * rc / channel.c
az_0 = 1531
s_0 = az_0 / speed
# wavelength
lamb = channel.c / channel.radar.fc
# range bandwidth
Br = channel.radar.pulse.get_bandwidth()
# doppler bandwidth
# azimuth swath
ant_az_l = rc * 2 * np.tan(lamb / (2 * channel.radar.antenna.L))
Bd = np.abs(fd_r(ant_az_l / (2 * speed), rc, speed, lamb) - fd_r(-ant_az_l / (2 * speed), rc, speed, lamb))
# light speed
c = channel.c
image = impulse_image(image,
                      s_0,
                      t_0,
                      lamb,
                      Br,
                      Bd,
                      speed,
                      c,
                      true_fast_time_ax,
                      slow_time_ax)

#%% plot image
fig, ax = plt.subplots(1)
c = ax.pcolormesh(fast_time_ax, slow_time_ax,np.abs(image),
                  shading='auto', cmap=plt.get_cmap('jet'))

fig.colorbar(c)
# ax.set_xlim((401e3, 412e3))
ax.set_xlabel("fast time [s]")
ax.set_ylabel("slow time [s]")

#%% pickle
with open('theorimage.pk', 'wb') as handle:
    pk.dump(image, handle)

#%% import copmpressed image
with open('theorimage.pk', 'rb') as handle:
    image = pk.load(handle)
    handle.close()
#%%
with open('data_dump.pk', 'rb') as handle:
    data = pk.load(handle)
    data.load_all()
    compimage = data.reconstructed_image
    handle.close()

#%% plot difference image
fig, ax = plt.subplots(1,2, sharex=True, sharey=True, constrained_layout = True)
c = ax[0].pcolormesh(fast_time_ax, slow_time_ax,(np.abs(image)),
                  shading='auto', cmap=plt.get_cmap('jet'), vmax=Bd)

ax[1].set_xlim((-0.35e-5+t_0 % (1/channel.radar.prf), 0.35e-5+t_0 % (1/channel.radar.prf)))
#ax[0].set_ylim((0.256,0.267))
ax[0].set_xlabel("fast time [s]")
ax[0].set_ylabel("slow time [s]")
ax[0].set_title("theoretical")
c1 = ax[1].pcolormesh(fast_time_ax, slow_time_ax,(np.abs(compimage)),
                  shading='auto', cmap=plt.get_cmap('jet'), vmax=Bd)

ax[1].set_xlim((-0.35e-5+t_0 % (1/channel.radar.prf), 0.35e-5+t_0 % (1/channel.radar.prf)))
ax[1].set_ylim((-0.004+s_0,0.004+s_0))
ax[1].set_xlabel("fast time [s]")
ax[1].set_ylabel("slow time [s]")
ax[1].set_title("range-doppler alg.")
fig.colorbar(c, ax=ax[1], shrink=0.99, location= 'right')

#%% plot difference image angle
fig, ax = plt.subplots(1,2, sharex=True, sharey=True, constrained_layout = True)
c = ax[0].pcolormesh(fast_time_ax, slow_time_ax,(np.angle(image)),
                  shading='auto', cmap=plt.get_cmap('jet'))#, vmax=Bd)

ax[1].set_xlim((-0.35e-5+t_0 % (1/channel.radar.prf), 0.35e-5+t_0 % (1/channel.radar.prf)))
#ax[0].set_ylim((0.256,0.267))
ax[0].set_xlabel("fast time [s]")
ax[0].set_ylabel("slow time [s]")
ax[0].set_title("theoretical")
c1 = ax[1].pcolormesh(fast_time_ax, slow_time_ax,(np.angle(compimage)),
                  shading='auto', cmap=plt.get_cmap('jet'))#, vmax=Bd)

ax[1].set_xlim((-0.35e-5+t_0 % (1/channel.radar.prf), 0.35e-5+t_0 % (1/channel.radar.prf)))
ax[1].set_ylim((-0.004+s_0,0.004+s_0))
ax[1].set_xlabel("fast time [s]")
ax[1].set_ylabel("slow time [s]")
ax[1].set_title("range-doppler alg.")
fig.colorbar(c, ax=ax[1], shrink=0.99, location= 'right')

#%% cut plots

t0_index = t_0 % (1 / channel.radar.prf)
print(t0_index)
t0_index = np.round(t0_index * data.Fs).astype('int')
s0_index = np.round((s_0-slow_time_ax[0]) * channel.radar.prf).astype('int')

fig, ax = plt.subplots(2,1)
ax[0].plot(data.get_range_axis(channel.c),np.abs(compimage[s0_index,:]),label='r-d alg.')
ax[0].plot(data.get_range_axis(channel.c),np.abs(image[s0_index,:]),label='theor.')
ax[0].set_xlim((channel.c/2)*(-.7e-5+t_0 % (1/channel.radar.prf)), (channel.c/2)*(.7e-5+t_0 % (1/channel.radar.prf)))
ax[0].legend(loc='upper right')
ax[0].set_xlabel('enveloped range [m]')
ax[1].plot(data.get_azimuth_axis(7.66e3), np.abs(compimage[:, t0_index]),label='r-d alg.')
ax[1].plot(data.get_azimuth_axis(7.66e3), np.abs(image[:, t0_index]),label='theor.')
ax[1].set_xlim((-0.004+s_0)*7.66e3,7.66e3*(0.004+s_0))
ax[1].set_xlabel('azimuth [m]')
ax[1].legend(loc='upper right')
ax[0].set_title('pulse orthogonal main cuts')
ax[0].grid()
ax[1].grid()
