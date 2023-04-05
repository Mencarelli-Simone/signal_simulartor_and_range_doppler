#  ____________________________Imports_____________________________
import numpy as np
import scipy.special as sc
from scipy import integrate
from linearFM import Chirp
from tqdm import tqdm
from dataMatrix import Data
from tqdm import tqdm


#  ____________________________Utilities __________________________

def matched_integrand_re(t1, t, instance):
    # private function integrand. the integral needs then to be multiplied by B
    return np.real(np.conjugate(instance.pulse.baseband_chirp(-t1)) * np.sinc(np.pi * instance.bandwidth * (t - t1)))


def matched_integrand_im(t1, t, instance):
    # private function integrand. the integral needs then to be multiplied by B
    return np.imag(np.conjugate(instance.pulse.baseband_chirp(-t1)) * np.sinc(np.pi * instance.bandwidth * (t - t1)))


# _____________________________Classes ____________________________
class MatchedFilter:
    """ Matched filter class describing a matched filter parameters
        and providing methods to apply the filter to a given signal
    """

    def __init__(self, pulse: Chirp):
        """
        Creates the matched filter for the given impulse shape
        :param pulse: Chirp class instance
        """
        self.pulse = pulse
        # if wanted the bandwidth could be different than the pulse one (at the expense of some performances)
        self.bandwidth = pulse.get_bandwidth()

    def fast_convolution(self, x, sampling_freq):
        """
        ifft( fft(x) * H(f) ) a circular convolution
        :param x signal to process
        :param sampling_freq sampling frequency of x
        :return filtered signal
        """
        # frequency axis creation
        f = np.linspace(-sampling_freq / 2, sampling_freq / 2 - sampling_freq / len(x), len(x))
        # TRUE SPECTRUM IMPLEMENTATION
        transfer_function = 1 / self.pulse.chirp_spectrum(f) * np.where(np.abs(f) <= self.pulse.get_bandwidth() / 2, 1, 0)
        transfer_function = np.fft.ifftshift(transfer_function)
        spectrum = np.fft.fft(np.fft.ifftshift(x)) / sampling_freq  # normalization #todo check if ifftshift is the correct one
        spectrum *= transfer_function
        # POSP IMPLEMENTATION
        #spectrum *= np.fft.ifftshift(self.pulse.chirp_matched_filter_posp(f))
        #spectrum *= np.fft.ifftshift(np.where(np.abs(f) <= self.pulse.get_bandwidth() / 2, 1, 0))
        convolved_signal = np.fft.ifft(spectrum)  * sampling_freq   # denormalization
        return np.fft.fftshift(convolved_signal), spectrum

    def zero_padded_fast_convolution(self, x, sampling_freq, alias_level):
        """
        ifft( fft(x) * H(f) ) a circular convolution
        :param x:               signal to process
        :param sampling_freq:   sampling frequency of x
        :param alias_level:     maximum allowable tail amplitude between 0 and 1
        :return y, spectra      filtered signal time and freq domain
        """
        # number of zero taps actually double of what is needed to achieve alias_level
        delta_zero = 1 / (alias_level * np.pi * self.pulse.get_bandwidth())
        # delta_zero = 10
        padding_sequence = 1j * np.zeros(int(delta_zero * sampling_freq + 1))
        # pad
        pad_x = np.concatenate((padding_sequence, x, padding_sequence))
        print('size of sequence increased by ', len(padding_sequence) * 2, 'samples ie ', delta_zero, ' s')
        pad_y, spectra = self.fast_convolution(pad_x, sampling_freq)
        y = pad_y[len(padding_sequence):len(padding_sequence) + len(x)]
        return y, spectra

    def fast_convolution_segmented(self, x, sampling_freq, segment_sample_size):
        """
        convolution in blocks
        :param x:
        :param sampling_freq:
        :param segment_sample_size:
        :return:
        """
        alias_level = 0.0001
        # number of zero taps actually double of what is needed to achieve alias_level
        delta_zero = 1 / (alias_level * np.pi * self.pulse.get_bandwidth())
        # delta_zero = 10
        padding_sequence = 1j * np.zeros(int(delta_zero * sampling_freq + 1))
        # array size
        tot_size = len(x)
        # number of segments
        seg_num = np.ceil(tot_size/segment_sample_size).astype('int')
        # output initialization
        y = np.zeros_like(x)
        # block processing
        print("segmented matched filtering progress:")
        for i in tqdm(range(int(seg_num))):
            # current segment
            x_i = x[int(i * segment_sample_size): int( min(tot_size, segment_sample_size * (i+1)))]
            # pad
            pad_x_i = np.concatenate((padding_sequence, x_i, padding_sequence))
            # convolve
            pad_y, spectra = self.fast_convolution(pad_x_i, sampling_freq)
            # min index in output vector
            min_idx = max(0, i * segment_sample_size - len(padding_sequence))
            # max index in output vector
            max_idx = min(tot_size, i * segment_sample_size + len(padding_sequence) + len(x_i))
            # min index in segment vector
            min_idx_i = max(0, len(padding_sequence) - i * segment_sample_size)
            # max index in segment vector
            max_idx_i = min(len(pad_x_i), min_idx_i + (max_idx-min_idx))
            # overlapp and add
            y[int(min_idx): int(max_idx)] += pad_y[int(min_idx_i): int(max_idx_i)]
            # TODO test this
        return y, np.zeros_like(y)




    def range_line_fast_convolution(self, data: Data, alias_level):
        delta_zero = int(np.floor(data.Fs / (alias_level * np.pi * self.pulse.get_bandwidth())))
        delta_zero = int(self.pulse.duration * data.Fs)
        filtered_output = 1j * np.zeros_like(data.data)
        print('performing matched filtering by range lines')
        for ii in tqdm(range(data.rows_num)):
            line = data.get_range_line(ii, delta_zero)
            padded_out, spectra = self.zero_padded_fast_convolution(line, data.Fs, alias_level)
            filtered_output[ii * data.columns_num: (ii + 1) * data.columns_num] = \
                padded_out[delta_zero:delta_zero + data.columns_num]
        return filtered_output

    ######## TEST STUFF ###########
    def spectra_error_closed_form(self, f, sampling_frequency, delay, replicas):
        """
        TEST FUNCTION
        spectral error of the range compressed i.e. de-chirped signal for a delay only return
        :param f:                   frequency axis
        :param sampling_frequency:  sampling frequency
        :param delay:               delay applied to the return
        :param replicas:            number of spectral replicas to be considered in the calculation of \
                                    the error half of it, same amount of replicas on the right and on the left
        :return:                    error in frequency domain
        """
        summa = 1j * np.zeros_like(f)
        for i in (range(1, replicas)):
            summa += self.pulse.chirp_spectrum(f - i * sampling_frequency) \
                     * np.exp(-1j * delay * 2 * np.pi * i * sampling_frequency)
            summa += self.pulse.chirp_spectrum(f + i * sampling_frequency) \
                     * np.exp(-1j * delay * 2 * np.pi * -i * sampling_frequency)
        summa /= self.pulse.chirp_spectrum(f)
        error = -np.exp(-1j * 2 * np.pi * self.pulse.fc) * \
                np.exp(-1j * 2 * np.pi * f * delay) * \
                summa
        return error

    def spectra_error_fft_form(self, f, sampling_frequency, delay, replicas):
        """
        TEST FUNCTION
        spectral error of the range compressed i.e. de-chirped signal for a delay only return
        :param f:                   frequency axis
        :param sampling_frequency:  sampling frequency
        :param delay:               delay applied to the return
        :param replicas:            number of spectral replicas to be considered in the calculation of \
                                    the error half of it, same amount of replicas on the right and on the left
        :return:                    error in frequency domain
        """
        summa_num = self.pulse.chirp_spectrum(f)
        summa_den = self.pulse.chirp_spectrum(f)
        for i in (range(1, replicas)):
            summa_num += self.pulse.chirp_spectrum(f - i * sampling_frequency) \
                         * np.exp(-1j * delay * 2 * np.pi * i * sampling_frequency)
            summa_num += self.pulse.chirp_spectrum(f + i * sampling_frequency) \
                         * np.exp(-1j * delay * 2 * np.pi * -i * sampling_frequency)
            summa_den += self.pulse.chirp_spectrum(f - i * sampling_frequency)
            summa_den += self.pulse.chirp_spectrum(f + i * sampling_frequency)
        summa = 1 - summa_num / summa_den
        error = np.exp(-1j * 2 * np.pi * self.pulse.fc) * \
                np.exp(-1j * 2 * np.pi * f * delay) * \
                summa
        return error

    def impulse_resolution_closed_form(self, t, delay):
        """
        TEST FUNCTION
        :param t:       time axis
        :param delay:   pulse delay
        :return:
        """
        B = self.pulse.get_bandwidth()
        print('bis', B)
        y = B * np.exp(-1j * self.pulse.fc * 2 * np.pi * delay) * \
            np.sin(np.pi * B * (t - delay)) / (np.pi * B * (t - delay))
        return y

    # ########## CONVOLUTION MATRIX IMPLEMENTATION (SLOW) ##########################################
    # def set_block_size(self, size):
    #     """
    #     sets the block size and creates the convolution matrix for the matched filter
    #     :param size: integer,, size of the block(s) to be filtered
    #     :return: null
    #     """
    #     self.size = size
    #     self.generate_filter_matrix()
    #
    # # too slow
    # def generate_filter_matrix(self):
    #     """
    #     generates the filter convolution matrix
    #     :return:
    #     """
    #     pass
    #
    # # too slow
    # def matched_filter_in_time(self, t):
    #     # private, convolution between sinc(band limit) and chirp inverted and conjugated signal
    #     re_integral = integrate.quad(matched_integrand_re, -self.pulse.duration / 2, self.pulse.duration / 2,
    #                                  args=(t, self))
    #     im_integral = integrate.quad(matched_integrand_im, -self.pulse.duration / 2, self.pulse.duration / 2,
    #                                  args=(t, self))
    #     return complex(self.pulse.get_bandwidth()) * (re_integral[0] + 1j * im_integral[0])
