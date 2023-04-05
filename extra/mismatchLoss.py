from linearFM import modified_fresnel
import numpy as np

def chirpSpectrum(f, rate, duration):
    alpha = np.abs(np.pi * rate) # alpha needs to be taken as absolute
    omega = np.pi * 2 * f
    S = np.sqrt(np.pi / (2 * np.abs(alpha))) * np.exp(-1j * omega ** 2 / (4 * alpha)) * (
            modified_fresnel((omega + np.abs(alpha) * duration) / (2 * np.sqrt(np.abs(alpha)))) - modified_fresnel(
        (omega - np.abs(alpha) * duration) / (2 * np.sqrt(np.abs(alpha)))))
    # the spectrum is calculated assuming an up chirp, if is a down chirp we apply the time reversal property of the
    # fourier transform
    S = np.where(k < 0, np.conjugate(S), S)
    return S

# The chirp spectrum can be used to find the pulse response given a mismatched filtering operation
# we are considering a unitary bandwidth B = 1 to normalize the resolution

# slow time axis
s = np.linspace(-8, 8, 10000)
# residual rate k = (k_signal - k_filter) / (k_signal * k_filter)
k = np.linspace(0, 20, 10000)

S, K = np.meshgrid(s, k)

# compressed slow time pulse approximation normalized:
g = chirpSpectrum(-S, K, 1)

# %% plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1)
c = ax.pcolormesh(K, S, np.abs(g))#10*np.log10(np.abs(g)))
ax.set_xlabel('residual rate')
ax.set_ylabel('normalized slow time')
fig.colorbar(c)
