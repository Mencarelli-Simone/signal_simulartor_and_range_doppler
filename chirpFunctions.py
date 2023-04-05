import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.special as sc

#matplotlib.use('TkAgg')


# bandwidth-time product and impulse duration
def tb2ktau(bandwidth, tb):
    duration = tb / bandwidth
    rate = bandwidth / duration
    return rate, duration


# real notation
def chirp(t, fc, rate, duration):
    return np.cos(2 * np.pi * (fc * t + rate * t ** 2 / 2)) * np.where(np.abs(t) <= duration / 2, 1, 0)


# baseband notation
def basebandChirp(t, rate, duration):
    # t = np.arange(-duration/2,duration/2,Ts) #  timespan double of the pulse
    s = 0.5 * np.exp(1j * np.pi * rate * t ** 2) * np.where(np.abs(t) <= duration / 2, 1, 0)
    return s


# fresnel test
def modified_fresnel(gamma):
    S, C = sc.fresnel(gamma * np.sqrt(2 / np.pi))
    return C + S * 1j


def chirpSpectrum(v, rate, duration):
    alpha = np.pi * rate
    omega = np.pi * 2 * v
    S = 0.5 * np.sqrt(np.pi / (2 * alpha)) * np.exp(-1j * omega ** 2 / (4 * alpha)) * (
            modified_fresnel((omega + alpha * duration) / (2 * np.sqrt(alpha))) - modified_fresnel(
        (omega - alpha * duration) / (2 * np.sqrt(alpha))))
    return S


def chirpSpectrumWithoutQuadraticPhase(v, rate, duration):
    alpha = np.pi * rate
    omega = np.pi * 2 * v
    S = 0.5 * np.sqrt(np.pi / (2 * alpha)) * (
            modified_fresnel((omega + alpha * duration) / (2 * np.sqrt(alpha))) - modified_fresnel(
        (omega - alpha * duration) / (2 * np.sqrt(alpha))))
    return S

def fresnelside(v,rate,duration):
    alpha = np.pi * rate
    omega = np.pi * 2 * v
    S = modified_fresnel((omega + alpha * duration) / (2 * np.sqrt(alpha)))
    return S


if __name__ == '__main__':
    bandwidth = 10  # Hz
    fc = 10  # Hz
    tb = 20
    rate, duration = tb2ktau(bandwidth, tb)
    print(rate, duration)

    t = np.linspace(-duration * 1 / 2, duration * 1 / 2, 4000000)
    s = chirp(t, fc, rate, duration)

    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(5)
    fig.set_figheight(5 * 3 / 4)
    ax.plot(t, s, 'k')
    ax.set_xlabel("time [s]")
    ax.set_ylabel("amplitude")
    fig.savefig('realchirp.eps', format='eps')

    # baseband chirp
    s = basebandChirp(t, rate, duration)
    fig, ax = plt.subplots(2, 1)
    fig.set_figwidth(5)
    fig.set_figheight(5)
    ax[0].plot(t, np.real(s), 'k')
    ax[1].plot(t, np.imag(s), '--k')
    ax[1].set_xlabel("time [s]")
    ax[0].set_ylabel("Real")
    ax[1].set_ylabel("Imaginary")
    fig.savefig('cplxchirp.eps', format='eps')

    # chirp spectrum

    f = np.linspace(-bandwidth, bandwidth, 100000)
    S = chirpSpectrum(f, rate, duration)
    Sa= chirpSpectrumWithoutQuadraticPhase(f, rate, duration)
    fig, ax = plt.subplots(2, 3)
    fig.set_figwidth(10)
    fig.set_figheight(5)
    ax[0,0].plot(f, np.abs(S), 'k')
    ax[1,0].plot(f, (np.angle(S)), 'k:')
    ax[1,0].plot(f, np.angle(Sa), 'k')
    ax[1,0].set_xlabel("freq [Hz]")
    ax[0,0].set_ylabel("Magnitude")
    ax[1,0].set_ylabel("... Phase             \n__ Residual Phase")
    ax[0,0].set_xlim((-bandwidth,bandwidth))
    ax[1,0].set_xlim((-bandwidth, bandwidth))
    ax[0, 0].set_title("$\\tau_p$B = 20 ")

    # chirp spectrum varying the bandwidth time product
    tb = 50
    rate, duration = tb2ktau(bandwidth, tb)
    print(rate, duration)
    S1 = chirpSpectrum(f, rate, duration)
    S1a = chirpSpectrumWithoutQuadraticPhase(f, rate, duration)
    ax[0, 1].plot(f, np.abs(S1), 'k')
    #ax[1, 1].plot(f, (np.angle(S1)), 'k:')
    ax[1, 1].plot(f, np.angle(S1a), 'k')
    ax[1, 1].set_xlabel("freq [Hz]")
    ax[0, 1].set_xlim((-bandwidth, bandwidth))
    ax[1, 1].set_xlim((-bandwidth, bandwidth))
    ax[0, 1].set_title("$\\tau_p$B = 50 ")

    tb = 150
    rate, duration = tb2ktau(bandwidth, tb)
    print(rate, duration)
    S2 = chirpSpectrum(f, rate, duration)
    S2a = chirpSpectrumWithoutQuadraticPhase(f, rate, duration)
    ax[0, 2].plot(f, np.abs(S2), 'k')
    #ax[1, 2].plot(f, (np.angle(S2)), 'k:')
    ax[1, 2].plot(f, np.angle(S2a), 'k')
    ax[1, 2].set_xlabel("freq [Hz]")
    ax[0, 2].set_xlim((-bandwidth, bandwidth))
    ax[1, 2].set_xlim((-bandwidth, bandwidth))
    ax[0, 2].set_title("$\\tau_p$B = 150 ")

    plt.show()
    a = input()
