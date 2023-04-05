#  ____________________________imports_____________________________
import numpy as np
from linearFM import Chirp
from geometryRadar import RadarGeometry
from pprint import pprint
from antenna import Antenna
#  ____________________________utilities __________________________


# from antenna import Antenna
# ___________________________________________ Classes ______________________________________________

class Radar:
    def __init__(self, path='pippo'):
        ## Geometric description of radar platform
        self.geometry = RadarGeometry()
        # radar initial position
        self.geometry.set_initial_position(0, 0, 408E3)
        # radar trajectory
        self.geometry.set_rotation(0, 0, 0)  # down looking and travelling along x axis
        # radar speed
        self.geometry.set_speed(7.66E3)  # m/s
        # carrier frequency
        self.fc = 10e9 # Hz
        ## Signals emitted from the radar
        # radar prf
        self.prf = 6e3 # Hz
        # transmitted impulse
        self.pulse = Chirp()
        # impulse setup
        self.pulse.set_kt_from_tb(2E6,200) # 2  MHz band, TB prod 200
        self.set_carrier_frequency(10E9) # 10GHz
        # radar antenna
        if path != 'pippo':
            self.antenna = Antenna(path=path)
        else:
            self.antenna = Antenna()
        # default initialization values print out
        # pprint('default values')
        # pprint(self.__dict__)
        # pprint('geometry: ')
        # pprint(self.geometry.__dict__)
        # pprint('pulse: ')
        # pprint(self.pulse.__dict__)

    def transmitted_signal(self, t):
        """
        compute the transmitted signal
        :param t: time axis
        :return: impulse train
        """
        return self.pulse.baseband_chirp_train_fast(t,self.prf)

    def set_prf(self, prf: float):
        self.prf = prf

    def set_carrier_frequency(self, fc):
        self.fc = fc
        self.pulse.fc = self.fc


# _____________________________Test________________________________________
if __name__ == '__main__':
    radar = Radar()