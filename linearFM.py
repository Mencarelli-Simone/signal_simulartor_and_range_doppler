#  ____________________________Imports_____________________________
import numba
import numpy as np
import scipy.special as sc
from numba import jit, prange
from tqdm import tqdm
#  ____________________________Utilities __________________________
# fresnel complexIntegral
from numpy import sign

def modified_fresnel(gamma):
    S, C = sc.fresnel(gamma * np.sqrt(2 / np.pi))
    return C + S * 1j


# a part of the chirp spectrum
def fresnelside(v, rate, duration):
    alpha = np.pi * rate
    omega = np.pi * 2 * v
    S = modified_fresnel((omega + alpha * duration) / (2 * np.sqrt(alpha)))
    return S


def fresnelotherside(v, rate, duration):
    alpha = np.pi * rate
    omega = np.pi * 2 * v
    S = modified_fresnel((omega - alpha * duration) / (2 * np.sqrt(alpha)))
    return S


# _____________________________Classes ____________________________

class Chirp:
    """ Chirp class describes a linear fm impulse and provides
    functions to calculate it over passed time/ frequency spans
    as a baseband low-pass complex signal
    """

    def __init__(self):
        self.rate = 5
        self.fc = 1E9
        self.duration = 2

    def set_rate(self, rate):
        """ frequency sweep rate
            impulse duration isn't varied
        """
        self.rate = rate

    def set_central_freq(self, fc):
        """ central frequency of the impulse
        """
        self.fc = fc

    def set_duration(self, duration):
        """ impulse duration in seconds
            chirp rate isn't varied
        """
        self.duration = duration

    def get_bandwidth(self):
        """ impulse approximate bandwidth (valid for high time bandwidth products)
        """
        return self.rate * self.duration

    def set_kt_from_tb(self, bandwidth, tb):
        """ set rate and duration expressed in term of bandwidth and
        duration-bandwidth product
        :param bandwidth:   approximate bandwidth of the signal or system available
                            bandwidth for tx-rx
        :param tb:          time-bandwidth product

        :return: rate, duration  set values
        """
        self.duration = tb / bandwidth
        self.rate = bandwidth / self.duration
        return self.rate, self.duration

    def chirp_spectrum(self, v):
        """ calculates the chirp spectrum over the passed frequency vector v
        :param  v:, frequency numpy ndarray
        :return:, complex base-banded spectrum
        """
        rate = self.rate
        duration = self.duration
        alpha = np.pi * rate
        omega = np.pi * 2 * v
        # Uncomment one of the following
        # -> basebanded 1W signal with 1/sqrt(2) amplitude
        # S = (1 / np.sqrt(2)) * np.sqrt(np.pi / (2 * np.abs(alpha))) * np.exp(-1j * omega ** 2 / (4 * alpha)) * (
        #         modified_fresnel((omega + alpha * duration) / (2 * np.sqrt(np.abs(alpha)))) - modified_fresnel(
        #     (omega - alpha * duration) / (2 * np.sqrt(np.abs(alpha)))))
        # -> normalized baseband signal with 1 amplitude
        S = np.sqrt(np.pi / (2 * np.abs(alpha))) * np.exp(-1j * omega ** 2 / (4 * alpha)) * (
                modified_fresnel((omega + alpha * duration) / (2 * np.sqrt(np.abs(alpha)))) - modified_fresnel(
            (omega - alpha * duration) / (2 * np.sqrt(np.abs(alpha)))))
        return S

    def chirp_matched_filter_posp(self, v):
        """ calculates the inverse chirp spectrum using the principle of stationary phase over the passed frequency vector v
        :param  v:, frequency numpy ndarray
        :return:, complex base-banded spectrum
        """
        rate = self.rate
        H = np.exp(-1j * np.pi / 4) * np.exp(1j * np.pi * v**2 / rate)
        return H

    def real_chirp(self, t):
        """
        chirp in real notation
        :param t: time axis, numpy ndarray
        :return: linear fm impulse calculated over t
        """
        fc = self.fc
        rate = self.rate
        duration = self.duration
        return np.cos(2 * np.pi * (fc * t + rate * t ** 2 / 2)) * np.where(np.abs(t) <= duration / 2, 1, 0)

    def baseband_chirp(self, t):
        """
        chirp in base-band notation (not delayed)
        :param t: time axis, numpy ndarray
        :return: linear fm basebanded impulse calculated over t
        """
        rate = self.rate
        duration = self.duration
        s = (1 / np.sqrt(2)) * np.exp(1j * np.pi * rate * t ** 2) * np.where(np.abs(t) <= duration / 2, 1, 0)
        return s

    def baseband_chirp_delayed(self, t, delay):
        """
        chirp in base-band notation (with delay)
        :param t: time axis, numpy ndarray
        :return: linear fm basebanded impulse calculated over t
        """
        rate = self.rate
        duration = self.duration
        fc = self.fc
        s = (1 / np.sqrt(2)) * np.exp(-1j * 2 * np.pi * fc * delay) * \
            np.exp(1j * np.pi * rate * (t - delay) ** 2) * \
            np.where(np.abs(t - delay) <= duration / 2, 1, 0)
        return s

    def baseband_chirp_train(self, t, prf):
        """
        chirp train in base-band notation i.e. sum of delayed versions of the chirp
        :param t: time axis
        :param prf: pulse repetition frequency
        :return: linear fm impulses train with one impulse centered on t = 0 and the others following
        """
        # find n_min minimum index for the impulse replica, rounded up
        n_min = np.ceil(t[0] * prf)
        # find the total number of replicas to be summed up
        n_tot = np.floor(((t[-1]-t[0]) * prf) + 1)

        print('n min = ', n_min, ' n_tot = ', n_tot)

        # initialize complex result vector
        s = 1j*np.zeros_like(t)

        # divide sum in segments
        # todo

        # sum up signal replicas
        for n in tqdm(range(int(n_min), int(n_min+n_tot))):
            s += self.baseband_chirp_delayed(t, n/prf)

        return s

    def baseband_chirp_train_fast(self, t, prf):
        """
        chirp train in base-band notation i.e. sum of delayed versions of the chirp
        :param t: time axis
        :param prf: pulse repetition frequency
        :return: linear fm impulses train with one impulse centered on t = 0 and the others following
        """
        # just for compilation
        pluto = self.basebanded_train_jit(t[0:2], prf, self.rate, self.duration, self.fc)
        # actual run
        print("computing basebanded chirp train with numba")
        return self.basebanded_train_jit(t, prf, self.rate, self.duration, self.fc)



    @staticmethod
    @jit(nopython = True, parallel = True)
    def basebanded_train_jit(t, prf, rate, duration, fc):
        # find n_min minimum index for the impulse replica, rounded up
        n_min = np.ceil(t[0] * prf)
        # find the total number of replicas to be summed up
        n_tot = np.floor(((t[-1] - t[0]) * prf) + 1)

        print('n min = ', n_min, ' n_tot = ', n_tot)

        # initialize complex result vector
        s = 1j * np.zeros_like(t)

        # sum up signal replicas
        for n in (prange(int(n_min), int(n_min + n_tot))):
            # divide sum in time segments
            # sum only the part between half pulse before and half pulse after ( pulses shall not overlap)
            impulse_begin_time = n / prf - .5 / prf
            impulse_end_time = n / prf + .5 / prf
            impulse_begin_index = (np.abs(t - impulse_begin_time)).argmin()
            impulse_end_index = (np.abs(t - impulse_end_time)).argmin()

            t1 = t[impulse_begin_index:impulse_end_index]
            delay = n / prf
            s1 = (1 / np.sqrt(2)) * np.exp(-1j * 2 * np.pi * fc * delay) * \
                 np.exp(1j * np.pi * rate * (t1 - delay) ** 2) * \
                 np.where(np.abs(t1 - delay) <= duration / 2, 1, 0)

            s[impulse_begin_index:impulse_end_index] += s1

        return s

    def baseband_chirp_train_fast1(self, t, prf):
        """
        chirp train in base-band notation i.e. sum of delayed versions of the chirp
        :param t: time axis
        :param prf: pulse repetition frequency
        :return: linear fm impulses train with one impulse centered on t = 0 and the others following
        """
        # find n_min minimum index for the impulse replica, rounded up
        n_min = np.ceil(t[0] * prf)
        # find the total number of replicas to be summed up
        n_tot = np.floor(((t[-1]-t[0]) * prf) + 1)

        print('n min = ', n_min, ' n_tot = ', n_tot)

        # initialize complex result vector
        s = 1j*np.zeros_like(t)

        # sum up signal replicas
        for n in tqdm(range(int(n_min), int(n_min+n_tot))):
            # divide sum in time segments
            # sum only the part between one pulse duration before and half pulse duration after (to account for distortions)
            impulse_begin_time = n / prf - self.duration
            impulse_end_time = n / prf + self.duration
            impulse_begin_index = (np.abs(t - impulse_begin_time)).argmin()
            impulse_end_index = (np.abs(t - impulse_end_time)).argmin()
            s[impulse_begin_index:impulse_end_index] += \
                self.baseband_chirp_delayed(t[impulse_begin_index:impulse_end_index], n/prf)

        return s

# jitted, just a wrapper
    def baseband_chirp_train_delayed(self, t, delay, prf):
        """
         chirp train arbitrarily delayed in base-band notation i.e. sum of delayed versions of the chirp
        :param t: time axis
        :param delay: delay( can be time dependent)
        :param prf: pulse repetition frequency
        :return: linear fm impulses train with one impulse centered on t = delay(0) and the others following
        """
        return self.baseband_chirp_train_delayed_jit(t, delay, prf, self.rate, self.duration, self.fc)


    @staticmethod
    @jit(nopython=True, parallel=True)
    def baseband_chirp_train_delayed_jit(t: np.ndarray, delay: np.ndarray, prf, rate, duration, fc):
        # allocate memory for rx signal
        x = 1j * np.zeros_like(t)
        # maximum delay
        max_delay = max(np.abs(delay))
        # minumum delay
        min_delay = min(np.abs(delay))
        # sampling period
        delta_t = t - np.roll(t, 1)
        Ts = np.sum(delta_t[1:len(delta_t)])/(len(delta_t)-1)
        # simpler way but less accurate
        # Ts = t[1] - t[0]
        # minimum tx pulse index
        # rational: t[0] - max_delay is the transmission time of the impulse received around t[0]
        # i.e. the first observed
        # the index of transmission is therefore found dividing by the pulse repetition interval i.e. x prf
        p_i_min = np.floor((t[0] - max_delay) * prf)
        # maximum tx pulse index # this might actually be out of the time axis. melius abundare quam deficere
        p_i_max = np.ceil((t[-1] - min_delay) * prf)
        # pulse transmission time axis
        p = np.arange(p_i_min, p_i_max) / prf
        # pulse destination time (based on the instantaneous delay at the time of transmission)
        p_d = np.zeros_like(p)

        print(" calculating pulse delays ")
        # delay indexes for pulse centers
        idx = (p - t[0]) / Ts
        # rounding (int casting done in the loop)
        idx = (0.5 + idx)
        # find the pulse destinations time
        # rational: this is needed to segment the time axis so that only one pri is taken in consideration
        # for the n-th impulse computation
        for ii in (prange(len(p_d))): # for every pulse
            # idx = np.argmin(np.abs(t - p[ii])) # pulse time axis index
            p_d[ii] = delay[int(idx[ii])] + p[ii] # pulse central delay in time
        # generate the delayed impulse
        print(" generating train of impulses")
        half_pri_samples_num = int(0.5 + 1 / (2 * prf * Ts)) # this should already be an integer for how the Ts and prf are chosen
        for ii in (prange(len(p))):
            # find current impulse index in the time axis
            low_idx = 0.5 + ((p_d[ii] - t[0])) / Ts - half_pri_samples_num
            low_idx = int(low_idx)
            if (low_idx < 0):
                low_idx = 0
            if (low_idx >= len(t)):
                low_idx = len(t) - 1
            high_idx = 0.5 + (p_d[ii] - t[0]) / Ts + half_pri_samples_num
            high_idx = int(high_idx)
            if (high_idx < 0):
                high_idx = 0
            if (high_idx >= len(t)):
                high_idx = len(t) - 1
            #print('index range',low_idx,' ', high_idx)

            t1 = t[low_idx:high_idx] # time axis around the impulse
            dlay = p[ii] + delay[low_idx:high_idx] # delay axis around the impulse
            # delayed impulse (time varying delay)
            sbb = s = 1 * np.exp(-1j * 2 * np.pi * fc * dlay) * \
                np.exp(1j * np.pi * rate * (t1 - dlay) ** 2) * \
                np.where(np.abs(t1 - dlay) <= duration / 2, 1, 0)
            # Anti aliasing processing
            # rational: the signal after being resampled in azimuth (slow time) is actually
            # a non baseband sampled signal, centered around fc, the carrier frequency. We can compensate for that
            # applying a slow time complex demodulation so that the azimuth fft is the one of a baseband signal.
            sbb *= np.exp(1j * 2 * np.pi * p[ii] * fc) # here the demodulation simply appears as a constant phase term

            # add the computed signal in the current time section to the complete raw signal
            x[low_idx:high_idx] += sbb
        print(" done generating train of impulses")
        return x












