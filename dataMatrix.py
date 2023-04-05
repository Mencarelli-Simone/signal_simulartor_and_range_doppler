#  ____________________________imports_____________________________
import os
import traceback

import numpy as np
import pickle as pk
from pointTarget import PointTarget
from linearFM import Chirp
from geometryRadar import RadarGeometry
from pprint import pprint


#  ____________________________utilities __________________________


# ___________________________________________ Classes ______________________________________________
class Data:
    """
    something that allows you to access the data throughout all the simulation line
    """

    def __init__(self, directory_path: str = "./Simulation_Data/"):
        # directory path
        self.directory_path = directory_path

        ## data storage template

        # parameters
        self.prf = None
        self.Fs = None
        self.rows_num = None
        self.columns_num = None

        # axis (these are filled at need when a getter is called - see getters section)
        self.fast_time_axis = None
        self.slow_time_axis = None
        self.range_axis = None
        self.azimuth_axis = None

        # arrays and matrices (filled at need - see setters section)
        self.time = None  # simulation time axis
        self.data = None  # raw signal array (not enveloped)
        self.data_matrix = None  # same of data, but enveloped on a 2-d array (slow and fast time)
        self.data_range_matrix = None  # range compressed signal in time domain - 2-d array
        self.doppler_range_compressed_matrix = None  # range compressed signal in the range-doppler domain - 2-d array
        self.doppler_range_compressed_matrix_rcmc = None  # range compressed signal in the range-doppler domain after
        # range cell migration compensation - 2-d array
        self.range_doppler_reconstructed_image = None  # azimuth filtered signal, just before ifft - 2-d array
        self.reconstructed_image = None  # final reconstructed image - 2-d array

    # SETTERS
    def set_directory_path(self, directory_path):
        self.directory_path = directory_path

    def set_prf(self, prf, Fs):
        """
        set the prf to access the data and return prf as a multiple of Fs
        :param prf:  pulse repetition frequency
        :return: none
        """
        pri = 1 / Fs * np.round(Fs / prf)
        self.prf = 1 / pri
        return self.prf

    def set_rx_data(self, data: np.ndarray):
        """
        store in memory the received radar signal for further processing
        :param data: complex basebanded return signal, must be of the same length of the time axis of this object
        :return: data reshaped in the form of a range/azimuth matrix
        """
        self.data = data
        if len(self.data) != len(self.time):
            print("ERROR, data must e computed on the provided time axis")
            return
        matrix_rows = int(np.round(len(self.data) / (self.Fs / self.prf)))
        self.rows_num = matrix_rows
        matrix_columns = int(np.round(len(self.data) / matrix_rows))
        self.columns_num = matrix_columns
        self.data_matrix = self.data.reshape((matrix_rows, matrix_columns))
        return self.data_matrix

    def set_range_compressed_data(self, data: np.ndarray):
        """
        store in memory the range-processed data
        :param data: complex-basebanded-range-compressed return signal
        :return: data reshaped in the form of a range/azimuth matrix
        """
        self.data_range = data
        if len(self.data_range) != len(self.time):
            print("ERROR, data must e computed on the provided time axis")
            return
        matrix_rows = int(np.round(len(self.data_range) / (self.Fs / self.prf)))
        matrix_columns = int(np.round(len(self.data_range) / matrix_rows))
        self.data_range_matrix = self.data_range.reshape((matrix_rows, matrix_columns))
        return self.data_range_matrix

    def set_doppler_range_compressed_matrix(self, data: np.ndarray):
        self.doppler_range_compressed_matrix = data

    def set_doppler_range_compressed_matrix_rcmc(self, data: np.ndarray):
        self.doppler_range_compressed_matrix_rcmc = data

    def set_range_doppler_reconstructed_image(self, data: np.ndarray):
        self.range_doppler_reconstructed_image = data

    def set_reconstructed_image(self, data: np.ndarray):
        self.reconstructed_image = data

    def time_axis(self, Fs, min, max):
        """
        creates a time axis aligned to the data matrix
        :param Fs: Sampling frequency
        :param min: minimum time
        :param max: maximum time
        :return: time axis
        """

        self.Fs = Fs
        min_ = int(np.floor(float(min) * self.prf)) / self.prf
        max_ = int(np.ceil(float(max) * self.prf)) / self.prf

        # make sure there is an odd number of rows in the image including the slow time = 0 if the limits are symmetric
        if int(((max_ - min_) * self.prf) % 2 ) == 0:
            max_ = int(1 + np.ceil(float(max) * self.prf)) / self.prf

        self.time = np.linspace(min_, max_ - 1 / self.Fs, np.round((max_ - min_) * self.Fs).astype('int'))
        if (len(self.time)) % np.int(0.5 + self.Fs / self.prf) != 0:
            print("ERRORROARRRRRR", (len(self.time)) % int(self.Fs / self.prf))
        return self.time

    # GETTERS

    def get_range_axis(self, c = 299792458):
        """
        creates the delay and range axis relative to a pulse repetition interval (not a true range, but an enveloped one)
        :param c: optional, propagation speed default is speed of light in vacuum
        :return: range axis
        """

        # relative range from delay within a pulse repetition interval
        self.range_axis = self.get_fast_time_axis() * c / 2
        return self.range_axis

    def get_fast_time_axis(self):
        """
        get the fast time axis (enveloped to pri)
        :return: fast time axis
        """
        # relative delay inside a pulse repetition interval
        self.fast_time_axis = np.linspace(0, (1 / self.prf) - (1 / self.Fs), int(np.round(self.Fs / self.prf)))
        return self.fast_time_axis

    def get_slow_time_axis(self):
        """
        get the slow time axis
        :return: slow time axis
        """
        matrix_time = self.time.reshape((self.rows_num, self.columns_num))
        self.slow_time_axis = matrix_time[:, 0].transpose()  # at row beginning
        return self.slow_time_axis

    def get_azimuth_axis(self, v):
        """
        creates the azimuth an slow time axes
        :param v: satellite absolute speed
        :return: azimuth axis
        """
        matrix_time = self.time.reshape((self.rows_num, self.columns_num))
        self.slow_time_axis = matrix_time[:, 0].transpose()  # at row beginning
        self.azimuth_axis = self.slow_time_axis[:] * v
        return self.azimuth_axis

    def get_doppler_axis(self, prf, doppler_centroid):
        """
        creates the doppler axis of the spectrum replica containing the
        Doppler centroid frequency, non enveloped around doppler centroid
        :param prf: Radar pulse repetition frequency
        :param doppler_centroid: Doppler centroid frequency
        :return: doppler axis
        """
        # same length of azimuth axis
        doppler_ax = np.linspace(-prf / 2,
                                 (1 - 1 / len(self.data_matrix[:, 0])) * prf / 2,
                                 len(self.data_matrix[:, 0]))
        f_offset = np.ceil(doppler_centroid / prf) * prf
        doppler_ax += f_offset
        return doppler_ax

    def get_range_line(self, index, extra_samples):
        """
        return a range line of raw data with a number of extra samples to the left and to the right
        :param index: range line index (row of data matrix)
        :param extra_samples: number of samples to return before and after the range line
        :return: range line with right an left data or zero padded tails
        """
        # check index validity
        if index < 0 or index >= len(self.data_matrix[0, :]):
            print("Error index out of boundaries")
            return
        ii = index * len(self.data_matrix[0, :])
        jj = ii - extra_samples
        # check if it needs zeros before
        zero_pad_left = 1j * np.zeros(int(-jj * (np.sign(-jj) + 1) / 2))
        jj += len(zero_pad_left)
        # check if it needs zeros after
        kk = ii + len(self.data_matrix[0, :]) + extra_samples
        zero_pad_right = 1j * np.zeros(int((kk - len(self.data)) *
                                           (np.sign(kk - len(self.data)) + 1) / 2))
        kk -= len(zero_pad_right)
        # pack everything
        data_out = self.data[jj:kk]
        data_out = np.concatenate((zero_pad_left, data_out, zero_pad_right))
        return data_out

    ## LOAD AND DUMP DATA FIELDS

    # DUMPERS
    def dump_rx_data(self, data: np.ndarray = None, dir_path: str = None):
        """
        dump the data array and the slow time wrapped matrix, frees the memory after dumping.
        can be used in stead of set rx_data but it will store the data directly on the hard disk.
        if no data array is passed it dumps the corresponding signal already stored as attribute
        :param data: optional arg, raw signal array to dump
        :param dir_path: optional arg, directory where to save the data. !! path SHALL end with '/' !!
        :return:
        """
        # check if data is passed as argument or if is already stored as object property
        if data != None:
            self.set_rx_data(data)

        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path

        # check if sub-folder exist, if not create it
        if not os.path.exists(path):
            os.makedirs(path)
        # pickle data
        filename = "Raw_Data"
        filename = path + filename + ".pk"
        with open(filename, 'wb') as handle:
            pk.dump(self.data, handle)
            handle.close()
        # pickle data_matrix
        filename = "Raw_Data_Matrix"
        filename = path + filename + ".pk"
        with open(filename, 'wb') as handle:
            pk.dump(self.data_matrix, handle)
            handle.close()
        # free memory
        self.data = None
        self.data_matrix = None
        print("Raw data and data matrix dumped")

    def dump_range_compressed_matrix(self, range_compressed_data: np.ndarray = None, dir_path: str = None):
        """
        dump the range compressed data matrix, frees the memory allocated for the corresponding attribute after dumping.
        can be used in stead of set_range_compressed_data but it will store the data directly on the hard disk.
        if no data array is passed it dumps the corresponding signal already stored as attribute
        :param range_compressed_data: optional arg, matrix to dump
        :param dir_path: optional arg, directory where to save the data. !! path SHALL end with '/' !!
        :return:
        """
        # check if data is passed as argument or if is already stored as an object attribute
        if range_compressed_data != None:
            self.set_range_compressed_data(range_compressed_data)

        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path

        # check if sub-folder exist, if not create it
        if not os.path.exists(path):
            os.makedirs(path)

        # pickle data
        filename = "Range_Compressed_Data"
        filename = path + filename + ".pk"
        with open(filename, 'wb') as handle:
            pk.dump(self.data_range_matrix, handle)
            handle.close()

        # free memory
        self.data_range_matrix = None
        print("Range compressed data matrix dumped")

    def dump_doppler_range_compressed_matrix(self, doppler_range_compressed_matrix: np.ndarray = None,
                                             dir_path: str = None):
        """
        dump the range-doppler transformed range compressed data matrix, 
        frees the memory allocated for the corresponding attribute after dumping.
        can be used in stead of set_doppler_range_compressed_data but it will store the data directly on the hard disk.
        if no data array is passed it dumps the corresponding signal already stored as attribute
        :param doppler_range_compressed_matrix: optional arg, matrix to dump
        :param dir_path: optional arg, directory where to save the data. !! path SHALL end with '/' !!
        :return:
        """
        # check if data is passed as argument or if is already stored as an object attribute
        if doppler_range_compressed_matrix != None:
            self.set_doppler_range_compressed_matrix(doppler_range_compressed_matrix)

        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path

        # check if sub-folder exist, if not create it
        if not os.path.exists(path):
            os.makedirs(path)

        # pickle data
        filename = "Doppler_Range_Compressed_Matrix"
        filename = path + filename + ".pk"
        with open(filename, 'wb') as handle:
            pk.dump(self.doppler_range_compressed_matrix, handle)
            handle.close()

        # free memory
        self.doppler_range_compressed_matrix = None
        print("Doppler-range range-compressed data matrix dumped")

    def dump_doppler_range_compressed_matrix_rcmc(self, doppler_range_compressed_matrix_rcmc: np.ndarray = None,
                                                  dir_path: str = None):
        """
        dump the range-doppler transformed range compressed data matrix, 
        frees the memory allocated for the corresponding attribute after dumping.
        can be used in stead of set_doppler_range_compressed_data_rcmc but it will store the data directly on the hard disk.
        if no data array is passed it dumps the corresponding signal already stored as attribute
        :param doppler_range_compressed_matrix_rcmc: optional arg, matrix to dump
        :param dir_path: optional arg, directory where to save the data. !! path SHALL end with '/' !!
        :return:
        """
        # check if data is passed as argument or if is already stored as an object attribute
        if doppler_range_compressed_matrix_rcmc != None:
            self.set_doppler_range_compressed_matrix_rcmc(doppler_range_compressed_matrix_rcmc)

        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path

        # check if sub-folder exist, if not create it
        if not os.path.exists(path):
            os.makedirs(path)

        # pickle data
        filename = "Doppler_Range_Compressed_Matrix_RCMC"
        filename = path + filename + ".pk"
        with open(filename, 'wb') as handle:
            pk.dump(self.doppler_range_compressed_matrix_rcmc, handle)
            handle.close()

        # free memory
        self.doppler_range_compressed_matrix_rcmc = None
        print("Doppler-range range-compressed RCMCed data matrix dumped")

    def dump_range_doppler_reconstructed_image(self, range_doppler_reconstructed_image: np.ndarray = None,
                                               dir_path: str = None):
        """
        dump the range-doppler azimuth and range filtered matrix, 
        frees the memory allocated for the corresponding attribute after dumping.
        can be used in stead of set_range_doppler_reconstructed_image but it will store the data directly on the hard disk.
        if no data array is passed it dumps the corresponding signal already stored as attribute
        :param range_doppler_reconstructed_image: optional arg, matrix to dump
        :param dir_path: optional arg, directory where to save the data. !! path SHALL end with '/' !!
        :return:
        """
        # check if data is passed as argument or if is already stored as an object attribute
        if range_doppler_reconstructed_image != None:
            self.set_range_doppler_reconstructed_image(range_doppler_reconstructed_image)

        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path

        # check if sub-folder exist, if not create it
        if not os.path.exists(path):
            os.makedirs(path)

        # pickle data
        filename = "Doppler_Range_Reconstructed_Image"
        filename = path + filename + ".pk"
        with open(filename, 'wb') as handle:
            pk.dump(self.range_doppler_reconstructed_image, handle)
            handle.close()

        # free memory
        self.range_doppler_reconstructed_image = None
        print("Doppler-range reconstructed image matrix dumped")

    def dump_reconstructed_image(self, reconstructed_image: np.ndarray = None, dir_path: str = None):
        """
        dump the reconstructed image, 
        frees the memory allocated for the corresponding attribute after dumping.
        can be used in stead of set_range_doppler_reconstructed_image but it will store the data directly on the hard disk.
        if no data array is passed it dumps the corresponding signal already stored as attribute
        :param reconstructed_image: optional arg, matrix to dump
        :param dir_path: optional arg, directory where to save the data. !! path SHALL end with '/' !!
        :return:
        """
        # check if data is passed as argument or if is already stored as an object attribute
        if reconstructed_image != None:
            self.set_reconstructed_image(reconstructed_image)

        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path

        # check if sub-folder exist, if not create it
        if not os.path.exists(path):
            os.makedirs(path)

        # pickle data
        filename = "Reconstructed_Image"
        filename = path + filename + ".pk"
        with open(filename, 'wb') as handle:
            pk.dump(self.reconstructed_image, handle)
            handle.close()

        # free memory
        self.reconstructed_image = None
        print("Reconstructed image matrix dumped")

    # LOADERS (specular to dumpers)
    def load_rx_data(self, dir_path: str = None):
        """
        loads rx raw data and rx raw data matrix from file, looking in the default directory unless specified otherwise
        :param dir_path: optional arg, directory where to load the data from. !! path SHALL end with '/' !!
        :return: False if couldn't load
        """
        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path
        # check if directory exists, if true load the data
        if os.path.exists(path):
            # data matrix
            filename = "Raw_Data_Matrix"
            filename = path + filename + ".pk"
            try:
                with open(filename, 'rb') as handle:
                    self.data_matrix = pk.load(handle)
                    handle.close()
                print(filename, " Loaded")
            except Exception as e:
                # some error
                # print(traceback.format_exc(e))
                print("Error trying to load File: ", filename)

            # data array
            filename = "Raw_Data"
            filename = path + filename + ".pk"
            try:
                with open(filename, 'rb') as handle:
                    self.data = pk.load(handle)
                    handle.close()
                print(filename, " Loaded")
            except Exception as e:
                # some error
                # print(traceback.format_exc(e))
                print("Error trying to load File: ", filename)
                return False
        else:
            print("Directory ", dir_path, " doesn't exist")
            return False

        return True

    def load_range_compressed_matrix(self, dir_path: str = None):
        """
        load range compressed data matrix from file, looking in the default directory unless specified otherwise
        :param dir_path: optional arg, directory where to load the data from. !! path SHALL end with '/' !!
        :return: False if couldn't load
        """
        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path
        # check if directory exists, if true load the data
        if os.path.exists(path):
            # data matrix
            filename = "Range_Compressed_Data"
            filename = path + filename + ".pk"
            try:
                with open(filename, 'rb') as handle:
                    self.data_range_matrix = pk.load(handle)
                    handle.close()
                print(filename, " Loaded")
            except Exception as e:
                # some error
                # print(traceback.format_exc(e))
                print("Error trying to load File: ", filename)
                return False
        else:
            print("Directory ", dir_path, " doesn't exist")
            return False

        return True

    def load_doppler_range_compressed_matrix(self, dir_path: str = None):
        """
        load range compressed range-doppler domain data matrix from file,
        looking in the default directory unless specified otherwise
        :param dir_path: optional arg, directory where to load the data from. !! path SHALL end with '/' !!
        :return: False if couldn't load
        """
        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path
        # check if directory exists, if true load the data
        if os.path.exists(path):
            # data matrix
            filename = "Doppler_Range_Compressed_Matrix"
            filename = path + filename + ".pk"
            try:
                with open(filename, 'rb') as handle:
                    self.doppler_range_compressed_matrix = pk.load(handle)
                    handle.close()
                print(filename, " Loaded")
            except Exception as e:
                # some error
                # print(traceback.format_exc(e))
                print("Error trying to load File: ", filename)
                return False
        else:
            print("Directory ", dir_path, " doesn't exist")
            return False

        return True

    def load_doppler_range_compressed_matrix_rcmc(self, dir_path: str = None):
        """
        load range compressed range-doppler domain RCMC-ed data matrix from file,
        looking in the default directory unless specified otherwise
        :param dir_path: optional arg, directory where to load the data from. !! path SHALL end with '/' !!
        :return: False if couldn't load
        """
        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path
        # check if directory exists, if true load the data
        if os.path.exists(path):
            # data matrix
            filename = "Doppler_Range_Compressed_Matrix_RCMC"
            filename = path + filename + ".pk"
            try:
                with open(filename, 'rb') as handle:
                    self.doppler_range_compressed_matrix_rcmc = pk.load(handle)
                    handle.close()
                print(filename, " Loaded")
            except Exception as e:
                # some error
                # print(traceback.format_exc(e))
                print("Error trying to load File: ", filename)
                return False
        else:
            print("Directory ", dir_path, " doesn't exist")
            return False

        return True

    def load_range_doppler_reconstructed_image(self, dir_path: str = None):
        """
        load reonstructed image range-doppler domain data matrix from file,
        looking in the default directory unless specified otherwise
        :param dir_path: optional arg, directory where to load the data from. !! path SHALL end with '/' !!
        :return: False if couldn't load
        """
        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path
        # check if directory exists, if true load the data
        if os.path.exists(path):
            # data matrix
            filename = "Doppler_Range_Reconstructed_Image"
            filename = path + filename + ".pk"
            try:
                with open(filename, 'rb') as handle:
                    self.range_doppler_reconstructed_image = pk.load(handle)
                    handle.close()
                print(filename, " Loaded")
            except Exception as e:
                # some error
                # print(traceback.format_exc(e))
                print("Error trying to load File: ", filename)
                return False
        else:
            print("Directory ", dir_path, " doesn't exist")
            return False

        return True

    def load_reconstructed_image(self, dir_path: str = None):
        """
        load reonstructed image range-doppler domain data matrix from file,
        looking in the default directory unless specified otherwise
        :param dir_path: optional arg, directory where to load the data from. !! path SHALL end with '/' !!
        :return: False if couldn't load
        """
        # check if directory path is the one passed or the default one
        if dir_path != None:
            path = dir_path
        else:
            path = self.directory_path
        # check if directory exists, if true load the data
        if os.path.exists(path):
            # data matrix
            filename = "Reconstructed_Image"
            filename = path + filename + ".pk"
            try:
                with open(filename, 'rb') as handle:
                    self.reconstructed_image = pk.load(handle)
                    handle.close()
                print(filename, " Loaded")
            except Exception as e:
                # some error
                # print(traceback.format_exc(e))
                print("Error trying to load File: ", filename)
                return False
        else:
            print("Directory ", dir_path, " doesn't exist")
            return False

        return True

    def load_all(self):
        """ try to load everything from the default filepath"""
        self.load_rx_data()
        self.load_range_compressed_matrix()
        self.load_doppler_range_compressed_matrix()
        self.load_doppler_range_compressed_matrix_rcmc()
        self.load_range_doppler_reconstructed_image()
        self.load_reconstructed_image()
        print("Data loading complete")
