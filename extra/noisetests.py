# attempting to simulate bandlimited noise
# %%
import numpy as np

# bandwidth
from tqdm import tqdm



def noise_bandlimit(osf):
    power_n = np.zeros(len(osf))
    power_bl_n = np.zeros(len(osf))
    B = 1.1E6  # Hz
    # duration
    duration = .05  # s
    for ii in tqdm(range(len(osf))):
        # %%
        # time axis
        sampnum = int(osf[ii] * B * duration)
        if sampnum % 2 == 0:
            sampnum += 1
        time = np.linspace(0, duration, sampnum)

        # noise vector
        noise = np.random.normal(0, np.sqrt(osf[ii]), len(time))

        # %%
        # powa
        power_n[ii] = np.sum(noise ** 2) / len(time)
        print(power_n[ii])

        # %% bandlimiting
        # spectrum
        noise_f = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(noise))) / (B * osf[ii])
        # frequency axis
        f = np.linspace(-(B * osf[ii]) / 2, (B * osf[ii]) / 2, len(time))
        bandlimited_noise_f = np.where(np.abs(f) <= B / 2, noise_f, 0)
        # in time
        bandlimited_noise = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(bandlimited_noise_f))) * (B * osf[ii])

        # %%
        # new power
        power_bl_n[ii] = np.sum(np.abs(bandlimited_noise) ** 2) / len(time)
        print(power_bl_n[ii])

    return power_bl_n, power_n

def azimuth_fft_power(signal_matrix, prf):
    # perform fft by columns and multiply by normalization factor to see what happens

    rows_spectrum = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(signal_matrix, axes=0), axis=0), axes=0) / prf
    columns_power = np.sum(np.abs(rows_spectrum)**2, axis=0) / len(rows_spectrum[:,0])
    return columns_power, rows_spectrum


if __name__ == '__main__':
    # %%
    #ossf = np.arange(1,91,5)
    ossf = np.array([10,16])

    powbln, pown = noise_bandlimit(ossf)
    #%%
    result = (powbln / pown) - (1 / ossf)

    # %%
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1)
    ax.plot(ossf, result)

    #%% data bandlimited
    samples = 1001*2001
    B = 1.1E6
    osf = 10
    Fs = B * osf
    time = np.linspace(0, samples/Fs, samples)
    noise = np.random.normal(0, np.sqrt(osf), len(time))
    noise_spec = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(noise))) / Fs
    freq = np.linspace(-Fs/2, Fs/2, len(time))
    noise_spec_bl = np.where(np.abs(freq) <= B/2, noise_spec, 0)
    noise_bl = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(noise_spec_bl))) * Fs
    power = np.sum(np.abs(noise_bl)**2)/len(time)
    print("bandlimited power ", power)
    # power over azimuth spectrum
    noise_bl = noise_bl.reshape((1001, 2001))
    pow_columns, rows_spec = azimuth_fft_power(noise_bl, Fs / 2001)
    expected = np.ones_like(pow_columns) * 1001 / (Fs / 2001)**2

    fig, ax = plt.subplots(1)
    ax.plot(pow_columns)
    ax.plot(expected)
    fig, ax = plt.subplots(1)
    ax.imshow(np.abs(rows_spec))

