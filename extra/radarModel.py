#  ____________________________imports_____________________________
import numpy as np
from linearFM import Chirp      # chirp class

#  ____________________________utilities __________________________

# polar to rectangular
def sph2cart(P) -> np.ndarray:
    """ Convert a numpy array in the form [r,theta,phi] to a numpy array in the form [x,y,z]
  """
    r, theta, phi = 0, 1, 2
    x = P[r] * np.sin(P[theta]) * np.cos(P[phi])
    y = P[r] * np.sin(P[theta]) * np.sin(P[phi])
    z = P[r] * np.cos(P[theta])
    return np.array((x, y, z))


# rectangular to polar
def cart2sph(P) -> np.ndarray:
    """ Convert a numpy array in the form [x,y,z] to a numpy array in the form [r,theta,phi]
  """
    x, y, z = 0, 1, 2
    r = np.sqrt(np.dot(P, P))
    # theta
    if r != 0:
        theta = np.arccos(P[z] / r)
        phi = np.arctan2(P[y], P[x])
    else:
        theta, phi = 0, 0
    return np.array((r, theta, phi))


def test():
    print("I'm a radar model")


# ___________________________________________ Classes ______________________________________________
class Antenna:
    """ containing a gain pattern in theta,phi coordinates"""

    # parametric rectangle
    def __init__(self):
        self.W = 0.2  # m
        self.L = 0.5  # m
        self.lamda = 3E8 / 10E9

    def idealGain(self, theta_phi) -> complex:
        theta_x = (self.lamda / (2 * self.L))
        theta_y = (self.lamda / (2 * self.W))
        C = sph2cart(np.array((1, theta_phi[0], theta_phi[1])))
        if np.abs(np.arctan(np.abs(C[0]) / C[2])) < theta_x and np.abs(np.arctan(np.abs(C[1]) / C[2])) < theta_y:
            return 4 * np.pi * self.L * self.W / self.lamda ** 2
        else:
            return 0

    def setL(self, L):
        self.L = L

    def setW(self, W):
        self.W = W

    def getGain(self, theta_phi):
        """
        Parameters: thetaPhi numpy array in the form of [theta,phi]

        Return: gain at given solid angle coordiantes (local reference system)
        """
        return self.idealGain(theta_phi)


class Radar:
    """ attributes to describe the sensor in this model"""

    # rotation around the radar trajectory respect to zenith in the range direction
    lookingAngle = 1 * np.pi / 180
    # right hand rotation of the trajectory about zenith axis
    trajectoryAngle = 0
    # object attribute describing the antenna
    antenna = Antenna()

    def __init__(self):
        self.setRotation(self.lookingAngle, self.trajectoryAngle)
        # default parameters
        self.altitude = 405E3  # km
        self.velocity = 7.66E3  # m/s
        # initial  position of the satellite i.e. t=0
        self.S_0 = np.array((0, 0, self.altitude))

    def setAltitude(self, altitude: int):
        """ note doesn't automatically update the time axis"""
        self.altitude = altitude
        # initial  position of the satellite i.e. t=0
        self.setInitialPosition(np.array((0, 0, self.altitude)))

    def setInitialPosition(self, P_0):
        self.S_0 = P_0
        self.altitude = P_0[2]

    def setRotation(self, beta: float, alpha: float):
        self.lookingAngle = beta
        self.trajectoryAngle = alpha
        """ sets the local coordinate transformation matrices for a given looking angle """
        x_s = np.array((np.cos(alpha), np.sin(alpha), 0))
        y_s = np.array((np.cos(beta) * np.sin(alpha), -np.cos(beta) * np.cos(alpha), np.sin(beta)))
        z_s = np.array((np.sin(beta) * np.sin(alpha), -np.sin(beta) * np.cos(alpha), -np.cos(beta)))
        # local basis matrix (transformation matrix between local and canonical basis)
        self.Bs2c = np.column_stack((x_s, y_s, z_s))
        # and vice versa
        self.Bc2s = self.Bs2c.T

    def time(self, mintime: int, maxtime: int, samples: int):
        """ creates the time axis and radar displacement """
        # time axis
        self.t = np.linspace(mintime, maxtime, samples)
        self.pos = np.zeros((self.t.shape[0], 3))
        for i in np.arange(self.t.shape[0]):
            self.pos[i, :] = self.S_0 + self.velocity * self.t[i] * self.Bc2s[0, :]
        # position of the radar in global coordinates at each point in time
        # self.pos = np.column_stack((x,y,z))
        print("position and time axes of the radar created")

    def getBroadsideOnGround(self) -> np.ndarray:
        """ get the antenna broadside projection position on ground
        :return p_bsg position of the broadside projection on ground as numpy array in the form [x, y, 0] for each point in time
        """
        r = -self.pos[:,2]/self.Bc2s[2][2] # distance from satellite to bs on ground projection
        # allocation of p_bsg vector
        self.p_bsg = np.zeros_like(self.pos)
        # intersection point
        self.p_bsg[:, 0] = self.pos[:, 0] + self.Bc2s[2, 0] * r
        self.p_bsg[:, 1] = self.pos[:, 1] + self.Bc2s[2, 1] * r
        return self.p_bsg

    def rangeGain(self, P):
        """ returns range, gain respect to time for a point scatterer at a given position

            :parameter P numpy array in the form [x,y,z]

            :returns (Range, Gain) numpy arrays showing range and antenna gain relative to the target in time
        """
        # local cartesian reference
        P_i = P - self.pos  # offset
        P_sc = np.zeros_like(P_i)  # malloc for local cartesian
        self.P_sp = np.zeros_like(P_i)  # malloc for local spherical
        self.Gain = np.zeros_like(self.t)
        for i in np.arange(self.pos.shape[0]):
            P_sc[i] = self.Bc2s @ P_i[i]  # change of basis
            # print(P_sc[i])
            self.P_sp[i] = cart2sph(P_sc[i][:])  # change of projection
            # antenna gain at given angle:
            self.Gain[i] = self.antenna.getGain(self.P_sp[i, 1:3])
        self.Range = self.P_sp[:, 0]
        # todo doppler shift function note: if we consider the position axis instead of time,
        # the function is independent from speed.
        return (self.Range, self.Gain)  # range and gain
