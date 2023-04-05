#  ____________________________imports_____________________________
import os
import tracemalloc
from functools import partial
from threading import Thread

import numpy as np
from numba import njit, jit, prange
from tqdm import tqdm

from dataMatrix import Data
from matchedFilter import MatchedFilter
from pointTarget import PointTarget, TargetData
from radar import Radar
from utils import sph2cart, cart2sph


# ___________________________________________ Classes ______________________________________________

class Channel:
    def __init__(self, radar: Radar):
        """
        Class containing both the radar and the targets.
        it provides methods to generate the raw signal back-scattered from the point targets
        according to the radar parameters.
        It must be interfaced with a Data class instance to get the time axis and save the simulation outputs.
        :param radar: Radar object to be used for the simulation
        """
        self.radar = radar  # radar object
        self.target = []  # empty list of targets
        self.targets_cnt = 0  # number of targets in the list
        self.c = 299792458.0  # m/s speed of light in the medium (changing this allows for accounting different
        # propagation media or mechanisms e.g. mechanical waves)

    def add_target(self, target: PointTarget):
        """
        add a target to the scene
        :param target: PointTarget object
        :return: nothing
        """
        self.target.append(target)
        # set the target id to be the index in the list
        target.id = self.targets_cnt
        self.targets_cnt += 1

    def get_relative_range(self, t, target_id: int = 0):
        """
        get the distance in time between target and radar
        :param t: time axis
        :param target_id: optional parameter, target index to be used in this calculation
        :return: range in time or list of ranges in time
        """
        if len(self.target) != 0:
            return self.radar.geometry.get_range(self.target[target_id].pos_gcs, t)
        else:
            print("ERROR NO TARGETS IN SCENE")
            return None

    def get_relative_delay(self, t, target_id: int = 0):
        range_ = self.get_relative_range(t, target_id=target_id)
        delay = 2.0 * range_ / self.c  
        return delay

    def get_free_space_attenuation(self, t, target_id: int = 0):
        range_ = self.get_relative_range(t, target_id=target_id)
        attenuation = 1 / (4 * np.pi * range_ ** 2)  
        return attenuation

    def get_delayed_normalized_return(self, t, target_id: int = 0):
        """
        compute the normalized radar return for the target(s)
        :param t: time axis
        :param target_id: optional parameter, target index to be used in this calculation
        :return: delayed and distorted impulse train
        """
        delay = self.get_relative_delay(t, target_id=target_id)
        s = self.radar.pulse.baseband_chirp_train_delayed(t, delay, self.radar.prf)  
        return s

    def get_normalized_gain_weighting(self, t, target_id: int = 0):
        """
        return the normalized complex gain in the direction of observation for a target in every point of time
        :param t: time axis
        :param target_id: optional parameter, target index to be used in this calculation
        :return: antenna normalized gain vector (complex)
        """
        position = self.radar.geometry.get_lcs_of_point(self.target[target_id].pos_gcs,
                                                        t)  
        return np.abs(self.radar.antenna.get_gain_at_lcs_point(
            position))  # this should be faster, it uses jit functions

    def raw_signal_generator(self, data: Data, t_min, t_max, save_targets_data: bool = True,
                             target_data_path: str = "./Target_Data/", osf=10):
        """
        generate the raw signal, note the impulse bandwidth has to be setted before
        :param data: data class to store stuff in
        :param t_min: minimum time for the simulation
        :param t_max: maximum time for the simulation
        :param save_targets_data: optional boolean, save the data or not
        :param target_data_path: optional, string, directory where to save
        the target data (!!path Shall end with a '/'!!)
        :param osf: optional, oversampling factor default 10
        :return: nothing
        """
        # set sampling frequency for an oversampling factor of 10
        osf = osf
        Fs = osf * self.radar.pulse.get_bandwidth()

        # make sure the prf stored in radar settings is aligned to the data prf
        self.radar.set_prf(data.set_prf(self.radar.prf, Fs))

        # 0. Generate time axis
        print("0/2. Creating time axis")
        t = data.time_axis(Fs, t_min, t_max)
        # initialize rx signal
        x_rx = 1j * np.zeros_like(t)

        # For every target 
        print("computing target returns")
        for t_idx in tqdm(range(self.targets_cnt)):
            # 1. Compute antenna weighting
            print("Target ", t_idx + 1, "/", self.targets_cnt, " - 1/3. Computing antenna weighting")
            weights = self.get_normalized_gain_weighting(t, target_id=t_idx)

            # 1.5. Omissis: add here free space attenuation weights, the snr model does not include time dependant range
            # it is instead approximated to the beam centre values
            # just computing the range here to be saved in target data
            range_ = self.get_relative_range(t, t_idx)

            # 2. Compute the received impulse train
            print("Target ", t_idx + 1, "/", self.targets_cnt, " - 2/3. Computing delayed pulses")
            train = self.get_delayed_normalized_return(t, target_id=t_idx)

            # 3. store target data
            if save_targets_data:
                # check if sub-folder exist, if not create it
                if not os.path.exists(target_data_path):
                    os.makedirs(target_data_path)
                # create data target object
                data_t = TargetData(self.target[t_idx],
                                    pulse_train=train,
                                    antenna_weights=weights,
                                    free_space_range=range_)
                # dump the target data
                data_t.dump(target_data_path)
                # free memory
                del data_t

            # 3. Store the received impulse train
            x_rx += weights * train
        data.set_rx_data(x_rx)
        # return time ax and raw signal
        return t, x_rx


    # def raw_signal_generator_single_target(self, data: Data, save_targets_data: bool = True,
    #                          target_data_path: str = "./Target_Data/", t_idx=0):
    #     """
    #     generate the raw signal, simplified version of raw_signal_generator
    #     :param data: data class to store stuff in, time axis has to be present inside data
    #     :param save_targets_data: optional boolean, save the data or not
    #     :param target_data_path: optional, string, directory where to save
    #     the target data (!!path Shall end with a '/'!!)
    #     :param t_idx: target index in the data.target list
    #     :return: nothing
    #     """
    #
    #     # 0. retrieve time axis
    #     t = data.time
    #     # initialize rx signal
    #     x_rx = 1j * np.zeros_like(t)
    #     print("computing target", t_idx," return")
    #     # 1. Compute antenna weighting
    #     print("Target ", t_idx + 1, "/", self.targets_cnt, " - 1/3. Computing antenna weighting")
    #     weights = self.get_normalized_gain_weighting(t, target_id=t_idx)
    #
    #     # 1.5. Omissis: add here free space attenuation weights, the snr model does not include time dependant range
    #     # it is instead approximated to the beam centre values
    #     # just computing the range here to be saved in target data
    #     range = self.get_relative_range(t, t_idx)
    #
    #     # 2. Compute the received impulse train (moved outside thread)
    #
    #     # 3. store target data
    #     if save_targets_data:
    #         # check if sub-folder exist, if not create it
    #         if not os.path.exists(target_data_path):
    #             os.makedirs(target_data_path)
    #         # create data target object
    #         data_t = TargetData(self.target[t_idx],
    #                             antenna_weights=weights,
    #                             free_space_range=range)
    #         # dump the target data
    #         data_t.dump(target_data_path)
    #         # free memory
    #         del data_t
    #     return t_idx


    # def raw_signal_generator_multithread(self, data: Data, t_min, t_max, save_targets_data: bool = True,
    #                          target_data_path: str = "./Target_Data/", osf=10):
    #     """
    #     generate the raw signal
    #     :param data: data class to store stuff in
    #     :param t_min: minimum time for the simulation
    #     :param t_max: maximum time for the simulation
    #     :param save_targets_data: optional boolean, save the data or not
    #     :param target_data_path: optional, string, directory where to save
    #     the target data (!!path Shall end with a '/'!!)
    #     :param osf: optional, oversampling factor default 10
    #     :return: nothing
    #     """
    #     # set sampling frequency for an oversampling factor of 10
    #     osf = osf
    #     Fs = osf * self.radar.pulse.get_bandwidth()
    #
    #     # make sure the prf stored in radar settings is aligned to the data prf
    #     self.radar.set_prf(data.set_prf(self.radar.prf, Fs))
    #
    #     # 0. Generate time axis
    #     print("0/3. Creating time axis")
    #     t = data.time_axis(Fs, t_min, t_max)
    #     # initialize rx signal
    #     x_rx = 1j * np.zeros_like(t)
    #
    #     # create threads list
    #     threads = []
    #     # fill threads list
    #     for t_id in range(self.targets_cnt):
    #         tread = Thread(target=self.raw_signal_generator_single_target, args=(data,), kwargs={'save_targets_data':True, 't_idx':t_id})
    #         threads.append(tread)
    #
    #     # start threads
    #     for tread in threads:
    #         tread.start()
    #
    #     # join threads
    #     for tread in threads:
    #         tread.join()
    #
    #     # Generate the rx signal
    #     for t_id in range(self.targets_cnt):
    #         # create a target_data object to load the signal
    #         t_dat = TargetData(self.target[t_id])
    #         # load antenna weights
    #         t_dat.load(target_data_path)
    #         # create impulse train
    #         # 2. Compute the received impulse train
    #         print("Target ", t_id + 1, "/", self.targets_cnt, " - 2/3. Computing delayed pulses")
    #         train = self.get_delayed_normalized_return(t, target_id=t_id)
    #         # update target file
    #         t_dat.pulse_train = train
    #         t_dat.dump(target_data_path)
    #         # sum it up to generate the raw signal
    #         x_rx += t_dat.pulse_train * np.abs(t_dat.antenna_weights)
    #         # todo add here optional range attenuation
    #         # free memory
    #         del t_dat
    #     print("3/3. Combining target results")
    #     # save x_rx in data class
    #     data.set_rx_data(x_rx)
    #     # return time ax and raw signal
    #     return t, x_rx

    def filter_raw_signal(self, data: Data):
        """
        perform time matched filtering
        :param data:
        :return:
        """
        # create filter object
        filter = MatchedFilter(self.radar.pulse)
        # filter signal
        print("Performing matched filter fast convolution")
        # the maximum segment size is set to be 2^22
        compdata, spec = filter.fast_convolution_segmented(data.data, data.Fs, int(2 ** 24))
        # the returned spectrum spec has no significance here (is the spectrum of the last segment processed)
        # store range compressed signal
        data.set_range_compressed_data(compdata)


######################################## SELF TEST #####################################
def main(bw):
    # memory profiling
    # memory tracing
    tracemalloc.start()
    print("initial memory occupation:")
    size, peak = tracemalloc.get_traced_memory()
    print("size: ", size / (1024 ** 2), " MB , peak: ", peak / (1024 ** 2), " MB")
    # this should make the file chaintest obsolete
    # create objects
    radar = Radar()
    # use cubic interpolation for antenna pattern
    radar.antenna.cubic = True
    data = Data()
    channel = Channel(radar)
    target = PointTarget()
    #target1 = PointTarget()

    ## settings
    # setting time bandwidth product and bandwidth for the linear FM impulse
    channel.radar.pulse.set_kt_from_tb(bw, 80)
    # setting the sampling frequency for the simulated signal
    Fs = 10 * radar.pulse.get_bandwidth()
    prf = data.set_prf(8E3, Fs)
    # the radar doesn't know what's the Fs (sampling freq), but the prf must be a multiple of a sampling period,
    # that's why the prf must first be aligned using data.set_prf (and it is also stored internally to data)
    radar.set_prf(prf)  # 1000 Hz pulse repetition frequency
    # forward squint of the antenna is zero, lateral is 0.04 rad, just enough to point sideways the beam footprint (
    # ideal pattern)
    radar.geometry.set_rotation(0.04, 0, 0)

    # target position
    target.set_position_gcs(0, -15e3, 0)  # position of the target in meters (xyz) note the radar looks to the
    # right toward the negative y direction
    #target1.set_position_gcs(1431, -16e3, 0)  # position of the target in meters (xyz) note the radar looks to the
    # right toward the negative y direction

    # add a target to scene
    channel.add_target(target)
    #channel.add_target(target1)

    ## raw data and compression
    # generating the received raw signal
    channel.raw_signal_generator(data, -1, 1)
    # applying the compression filter to the simulated signal
    channel.filter_raw_signal(data)

    # %% store the simulation data for further processing
    import pickle as pk

    print('pickling')
    with open('./Simulation_Data/channel_dump.pk', 'wb') as handle:
        pk.dump(channel, handle)
        handle.close()
    with open('./Simulation_Data/data_dump.pk', 'wb') as handle:
        pk.dump(data, handle)
        handle.close()

    # memory tracing
    print("final memory occupation:")
    size, peak = tracemalloc.get_traced_memory()
    print("size: ", size / (1024 ** 2), " MB , peak: ", peak / (1024 ** 2), " MB")
    tracemalloc.stop()
    return str(("size: ", size / (1024 ** 2), " MB , peak: ", peak / (1024 ** 2), " MB"))


if __name__ == '__main__':
    main(2e6)
    # Bw = [2e6,5e6,10e6,30e6]
    # list = []
    # for bw in Bw:
    #     list.append(main(bw))
    # print(list)
