#  ____________________________imports_____________________________
import os

import numpy as np
import pickle as pk

# ___________________________________________ Classes ______________________________________________
from radar import Radar


class PointTarget:
    def __init__(self, index: int = 0):
        """
        point scatterer object creator
        :param index:   optional argument, default is 0
                        if different than 0 it means the target is part of a list
        """
        self.id = index
        self.rcs = 1
        self.pos_gcs = np.array([0, 0, 0])

    def set_position_gcs(self, x, y, z):
        self.pos_gcs = np.array([x, y, z])

    def set_position_lcs(self, radar: Radar, x_s: float, y_s: float, z_s: float):
        """ lcs with respect to the satellite origin point (t=0)
        """
        P = np.array([[x_s],
                      [y_s],
                      [z_s]])
        # local to global conversion
        P_lcs = np.matmul(radar.geometry.Bs2c, P) + radar.geometry.S_0
        # setting
        self.set_position_gcs(P_lcs[0, 0], P_lcs[1, 0], P_lcs[2, 0])

    def set_position_range_azimuth(self, radar: Radar, rng: float, az: float):
        """ considering lcs broadside """
        s_c = radar.geometry.velocity * az / radar.geometry.abs_v + radar.geometry.S_0
        P = s_c + rng * radar.geometry.z_s.reshape(3, 1)
        self.set_position_gcs(P[0, 0], P[1, 0], P[2, 0])


class TargetData:
    """
    companion class for PointTarget to pickle / unpickle target-relative data
    """

    def __init__(self,
                 target: PointTarget,
                 antenna_weights: np.ndarray = None,
                 pulse_train: np.ndarray = None,
                 free_space_range: np.ndarray = None):
        """
        This class provides a structure to serialize target relative data.
        It also provides methods to save itself into a file and load simulation data for a target identified by the
        target id field of the target object used to create an instance of this class
        :param target: PointTarget object
        :param antenna_weights: optional argument, antenna pattern track array
        :param pulse_train: optional argument, received normalized impulse pattern
        :param free_space_range: optional argument, distance between target and satellite at any point in time
        """
        self.target = target
        self.antenna_weights = antenna_weights
        self.pulse_train = pulse_train
        self.free_space_range = free_space_range

    def dump(self, file_path: str):
        """
        pickles itself inside the directory specified by the file path
        adding an unique suffix based on the target id field
        :param file_path: file path eg '\Docs\DataDir\' !! The last '\' Shall be included !!
        :return: None
        """
        index = str(int(self.target.id))
        # check if sub-folder exist, if not create it
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        filename = "TargetData_"
        filename = file_path + filename + index + ".pk"
        with open(filename, 'wb') as handle:
            pk.dump(self, handle)
            handle.close()
        print("target data ", index, " saved")

    def load(self, file_path: str):
        """
        load the data corresponding to the target used to initialize this object
        the data is loaded looking in the provided file path.
        :param file_path:
        :return: False if file not found
        """
        index = str(int(self.target.id))
        filename = "TargetData_"
        filename = file_path + filename + index + ".pk"
        try:
            with open(filename, 'rb') as handle:
                target_data = pk.load(handle)
                handle.close()
        except:
            print("ERROR: File "+ filename +" not loaded \ntarget data left empty")
            return False
        # copy all attributes in the current object
        self.target = target_data.target
        self.pulse_train = target_data.pulse_train
        self.free_space_range = target_data.free_space_range
        self.antenna_weights = target_data.antenna_weights
        print("target data ", index, " loaded")
        return True

# todo some testing to pickle/unpickle targets
################################################ SELF TEST ###################################################

if __name__ == '__main__':
    # testing nominal behavior
    # 1 create a target
    target = PointTarget(index=5)
    # 2 create an array
    array = np.arange(0,10)
    # 3 create a data target
    data = TargetData(target, free_space_range=array)
    # 4 dump it somewhere
    data.dump("./")
    # 5 create a new data target based on the same target
    data1 = TargetData(target)
    # 6 try to load saved data
    # target.id = 6
    data1.load("./")
    # 7 print and compare
    print(data.__dict__)
    print(data1.__dict__)