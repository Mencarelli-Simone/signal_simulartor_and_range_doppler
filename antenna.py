#  ____________________________imports_____________________________
import numpy as np
from numba import jit, prange
import os
import pickle as pk

from tqdm import tqdm

from interpolator_v2 import sphere_interp
from utils import sph2cart, cart2sph


#  ____________________________functions __________________________
@jit(nopython=True)
def fastIdealGain(lamda, L, W, theta_phi: np.ndarray) -> complex:
    """
    just in time compiled version of ideal gain function commented above
    :param lamda:
    :param L:
    :param W:
    :param theta_phi:
    :return:
    """
    forward_squint = 0.0
    theta_x = (lamda / (2 * L))
    theta_y = (lamda / (2 * W))
    C = sph2cart(np.array((1, theta_phi[0], theta_phi[1])))
    ff = np.sin(forward_squint)
    if np.abs(np.arctan(np.abs(C[0] - ff) / C[2])) < theta_x and np.abs(np.arctan(np.abs(C[1]) / C[2])) < theta_y:
        return 4 * np.pi * L * W / lamda ** 2
    else:
        return 1j * 0


# idealPatternCreator function
@jit(nopython=True, parallel=True)
def ideal_pattern_creator(lamda, L, W, theta_ax, phi_ax) -> np.ndarray:
    """
    creates a matrix containing a sampled version of the ideal square gain pattern
    :param lamda:
    :param L:
    :param W:
    :param theta_ax:
    :param phi_ax:
    :return: complex matrix containing gain pattern in theta and phi
    """
    gain_matrix = 1j * np.zeros((len(theta_ax), len(phi_ax)))
    for ii in prange(len(theta_ax)):
        for jj in range(len(phi_ax)):
            thetaphi = np.array([theta_ax[ii], phi_ax[jj]])
            gain_matrix[ii, jj] = fastIdealGain(lamda, L, W, thetaphi)
    return gain_matrix

@jit(nopython=True, parallel=True) # incredibly it is slower than the serial one 5-10 times slower. don't use this
def cubic_interpolation_parallel(theta_phi, Theta, Phi, gain, cubic: bool, gain_track: np.ndarray):
    """
    returns the gain interpolated along the track described by the thetha_phi couples coordinates
    :param theta_phi:   theta_phi[0,i] -> theta coordinate of i-th point in the track,
                        theta_phi[1,i] -> phi coordinate of i-th point in the track
    :param Theta:       Theta axis of the gain pattern (MUST BE HOMOGENEOUSLY SAMPLED)
    :param Phi:         Phi axis of the gain pattern (MUST BE HOMOGENEOUSLY SAMPLED)
    :param gain:        gain matrix sampled at Theta and Phi values
    :param cubic:       boolean, if true use cubic interpolation, if false use linear interpolation
    :param gain_track:  output gain vector
    :return:            gain weights along the track
    """
    # divide gain_track and theta_phi in subvectors
    threads = 4
    sublen = np.floor(len(theta_phi[0,:]) / threads)
    start_idx = (np.arange(0,int(threads)) * int(sublen)).astype(np.int64)
    end_idx = (start_idx + sublen).astype(np.int64)
    end_idx[-1] = len(theta_phi[0,:])
    end_idx = end_idx.astype(np.int64)
    print("attempting to create ", threads," threads")
    # compilation run
    t_p = theta_phi[:, 0:20]
    g_t = 1j * np.zeros_like(gain_track[0:20]).astype(np.complex128)
    g_t = cubic_interpolation(t_p, Theta, Phi, gain, cubic, g_t).astype(np.complex128)

    for pp in prange(threads): # it results 10 times slower than the serial version without any apparent reason
        t_p = theta_phi[:, int(start_idx[pp]):int(end_idx[pp])]
        g_t = 1j * np.zeros_like(gain_track[start_idx[pp]:end_idx[pp]]).astype(np.complex128)
        g_t = cubic_interpolation(t_p, Theta, Phi, gain, cubic, g_t).astype(np.complex128)
        gain_track[start_idx[pp]:end_idx[pp]] = g_t
    return gain_track



# interpolator function to be used in retrieving antenna weights
@jit(nopython=True)
def cubic_interpolation(theta_phi, Theta, Phi, gain, cubic: bool, gain_track: np.ndarray):
    """
    returns the gain interpolated along the track described by the thetha_phi couples coordinates
    :param theta_phi:   theta_phi[0,i] -> theta coordinate of i-th point in the track,
                        theta_phi[1,i] -> phi coordinate of i-th point in the track
    :param Theta:       Theta axis of the gain pattern (MUST BE HOMOGENEOUSLY SAMPLED)
    :param Phi:         Phi axis of the gain pattern (MUST BE HOMOGENEOUSLY SAMPLED)
    :param gain:        gain matrix sampled at Theta and Phi values
    :param cubic:       boolean, if true use cubic interpolation, if false use linear interpolation
    :param gain_track:  output gain vector
    :return:            gain weights along the track
    """
    delta_Theta = np.abs(Theta[1] - Theta[0])
    delta_Phi = np.abs(Phi[1] - Phi[0])

    # making sure phi is enveloped to 2 pi
    theta_phi[1] = theta_phi[1] % (2 * np.pi)
    theta_phi[0] = theta_phi[0] % (2 * np.pi)

    samples = len(gain_track)

    theta_min = min(Theta)
    theta_max = max(Theta)
    phi_min = min(Phi)
    phi_max = max(Phi)
    print('cubic ', cubic)

    # for every point in the track
    print(samples)
    for ii in range(samples):
        #print(ii,'of',len(gain_track) )
        # find 2 points before and after in theta (the data has even symmetry around pos/neg theta)
        theta_iidx = int(theta_phi[0, ii] / delta_Theta)
        theta_4idx = np.arange(theta_iidx - 1, theta_iidx + 3).astype(np.int64)
        # find 2 points before and after in phi (phi is circular in radians)
        phi_ii = int(theta_phi[1, ii] / delta_Phi) * delta_Phi
        phi_iidx = int((theta_phi[1, ii] % (2 * np.pi)) / delta_Phi)
        phi_4idx = np.linspace(phi_ii - delta_Phi, phi_ii + delta_Phi * 2, 4) % (2 * np.pi)
        # rounding
        phi_4idx = (0.5 + (phi_4idx / delta_Phi))
        # recasting
        phi_4idx = phi_4idx.astype(np.int64)
        # phi index specular about emisphere symmetry axis
        # specular_phi_4idx = np.round(((delta_phi * phi_4idx + np.pi) % (2 * np.pi)) / delta_phi) #round doesn't work in numba
        # these indexes will be needed later in case a negative theta value is encountered
        specular_phi_4idx = ((delta_Phi * phi_4idx + np.pi) % (2 * np.pi)) / delta_Phi
        specular_phi_4idx = np.where(specular_phi_4idx > 0, .5 + specular_phi_4idx, -.5 + specular_phi_4idx)
        specular_phi_4idx = specular_phi_4idx.astype(np.int64)
        # check out of boundaries index iidx
        if theta_iidx * delta_Theta < theta_min or theta_iidx * delta_Theta > theta_max or \
                phi_iidx * delta_Phi < phi_min or phi_iidx * delta_Phi > phi_max:
            print('ERROR: index out of boundaries impossible to interpolate')
            print(theta_phi[:, ii])
            return

        # adjust indexes for boundary cases (retain first and last samples)
        if np.abs(theta_4idx[0] * delta_Theta) < theta_min:
            theta_4idx[0] = theta_4idx[1]
        if np.abs(theta_4idx[3] * delta_Theta) > theta_max:
            theta_4idx[3] = theta_4idx[2]
        if phi_4idx[0] * delta_Phi < phi_min:
            phi_4idx[0] = phi_4idx[1]
        if phi_4idx[3] * delta_Phi > phi_max:
            phi_4idx[3] = phi_4idx[2]

        # phi_4idx, and theta_4idx are the indexes of the 4x4 interpolation matrix

        # allocate mem for 4 points of external interpolation interpolated at the desired (interpolated) theta
        # coordinate at different phi values
        phi_4 = 1j * np.zeros(4)
        for jj in range(4):
            # interpolate in theta dimension

            # if a theta index is negative, the corresponding phi coordinate shall be shifted by pi rad
            p_re = np.zeros(4)
            p_im = np.zeros(4)
            # fill p values ( interpolation points )
            for kk in range(4):
                # negative theta index case
                ph_i = phi_4idx[jj]
                if theta_4idx[kk] < 0:
                    ph_i = specular_phi_4idx[jj]
                th_i = int(np.abs(theta_4idx[kk]))
                # greater than 180´ theta index case
                if np.abs(theta_4idx[kk] * delta_Theta) > np.pi:
                    th_i = int(np.floor(np.pi / delta_Theta) - theta_4idx[kk])
                    ph_i = specular_phi_4idx[jj] #todo check phi index
                # fill p_re
                p_re[kk] = np.real(gain[th_i, ph_i])
                # fill p_im
                p_im[kk] = np.imag(gain[th_i, ph_i])

            # perform the cubic interpolation ( Catmull-Rom spline) to retrieve the gain value at the desired theta
            # coordinate

            # x must range between 0 and 1 ( inter-sample space)
            x = (theta_phi[0, ii] - theta_iidx * delta_Theta) / delta_Theta
            g_ph_re = 0.0
            g_ph_im = 0.0
            if cubic:
                g_ph_re = p_re[1] + 0.5 * x * (p_re[2] - p_re[0] + x * (
                        2.0 * p_re[0] - 5.0 * p_re[1] + 4.0 * p_re[2] - p_re[3] + x * (
                        3.0 * (p_re[1] - p_re[2]) + p_re[3] - p_re[0])))

                g_ph_im = p_im[1] + 0.5 * x * (p_im[2] - p_im[0] + x * (
                        2.0 * p_im[0] - 5.0 * p_im[1] + 4.0 * p_im[2] - p_im[3] + x * (
                        3.0 * (p_im[1] - p_im[2]) + p_im[3] - p_im[0])))
            else:
                g_ph_re = p_re[1] + (p_re[2] - p_re[1]) * x
                g_ph_im = p_im[1] + (p_im[2] - p_im[1]) * x
            phi_4[jj] = g_ph_re + 1j * g_ph_im

        # interpolate in phi dimension
        p_re = np.real(phi_4)
        p_im = np.imag(phi_4)
        x = ((theta_phi[1, ii] - phi_iidx * delta_Phi) % (2*np.pi)) / delta_Phi

        g_re = 0.0
        g_im = 0.0
        if cubic:
            g_re = p_re[1] + 0.5 * x * (p_re[2] - p_re[0] + x * (
                    2.0 * p_re[0] - 5.0 * p_re[1] + 4.0 * p_re[2] - p_re[3] + x * (
                    3.0 * (p_re[1] - p_re[2]) + p_re[3] - p_re[0])))

            g_im = p_im[1] + 0.5 * x * (p_im[2] - p_im[0] + x * (
                    2.0 * p_im[0] - 5.0 * p_im[1] + 4.0 * p_im[2] - p_im[3] + x * (
                    3.0 * (p_im[1] - p_im[2]) + p_im[3] - p_im[0])))
        else:
            g_re = p_re[1] + (p_re[2] - p_re[1]) * x
            g_im = p_im[1] + (p_im[2] - p_im[1]) * x

        gain_track[ii] = g_re + 1j * g_im


    return gain_track


# ___________________________________________ Classes ______________________________________________
class Pattern:
    """ structure to pickle and unpickle antenna patterns """

    def __init__(self, theta_ax, phi_ax, gain_pattern):
        self.gain_pattern = gain_pattern
        self.theta_ax = theta_ax
        self.phi_ax = phi_ax

    def load(self):
        """
        use this to copy at once the content
        :return: theta_ax, phi_ax, gain_pattern
        """
        return self.theta_ax, self.phi_ax, self.gain_pattern


class Antenna:
    """ containing a gain pattern in theta,phi coordinates"""

    # parametric rectangle
    def __init__(self, path='./Antenna_Pattern') :
        self.W = .8  # m
        self.L = 4  # m
        self.lamda = 3E8 / 10E9
        self.cubic = True  # interpolation method for antenna pattern, true cubic, false linear
        # ideal case boradside gain
        self.broadside_gain = 4 * np.pi * self.L * self.W / self.lamda ** 2
        # empty pattern
        self.theta_ax = None
        self.phi_ax = None
        self.gain_matrix = None

        # check if an antenna pattern exist, if not create an ideal one:
        pattern_folder_path = path
        pattern_file_name = 'gain_pattern.pk'
        # 1. check if subfolder exist, if not create it
        if not os.path.exists(pattern_folder_path):
            os.makedirs(pattern_folder_path)
        # 2. check if pattern exists in the folder, if not generate it
        if not os.path.exists(pattern_folder_path + '/' + pattern_file_name):
            # generate pattern
            print("no antenna pattern found, creating one")
            self.create_ideal_gain_pattern()
            # save pattern to file
            print("pickling antenna pattern")
            self.save_gain_pattern(pattern_folder_path + '/' + pattern_file_name)
        else:
            # 3. load the gain pattern as a matrix along with theta and phi axes
            print("loading antenna pattern")
            self.load_gain_pattern(pattern_folder_path + '/' + pattern_file_name)

        # compilation run for gain pattern
        # theta_phi = np.array([[0.1, 0.1], [0.2, 0.3]])
        # gain_track = 1j * np.zeros_like(theta_phi[0])
        # gain_track = cubic_interpolation(theta_phi, self.theta_ax, self.phi_ax, self.gain_matrix, self.cubic, gain_track)
        print("antenna initialized")


    def idealGain(self, theta_phi: np.ndarray) -> complex:
        gain_track = 1j * np.zeros_like(theta_phi[0])
        for ii in tqdm(range(len(gain_track))):
            gain_track[ii] = fastIdealGain(self.lamda, self.L, self.W, theta_phi[:,ii])
        return gain_track

    def create_ideal_gain_pattern(self):
        # theta sampling angle
        delta_theta = .06 * np.pi / 180
        # phi sampling angle
        delta_phi = .02 * np.pi / 180

        # polar angle axis
        self.theta_ax = np.linspace(0, np.pi - delta_theta, int(0.5 + np.pi / delta_theta))
        # azimuthal angle axis
        self.phi_ax = np.linspace(0, 2 * np.pi, int(0.5 + 2 * np.pi / delta_phi))
        self.gain_matrix = ideal_pattern_creator(self.lamda, self.L, self.W, self.theta_ax, self.phi_ax)

    def load_gain_pattern(self, filename):
        # unpickle pattern
        with open(filename, 'rb') as handle:
            pattern: Pattern = pk.load(handle)
            handle.close()
        # load into antenna
        self.theta_ax, self.phi_ax, self.gain_matrix = pattern.load()
        # delete pattern object
        del pattern
        pass

    def save_gain_pattern(self, filename):
        pattern = Pattern(self.theta_ax, self.phi_ax, self.gain_matrix)
        # pickle pattern
        with open(filename, 'wb') as handle:
            pk.dump(pattern, handle)
            handle.close()
        # delete pattern object
        del pattern


    def set_L(self, L):
        self.L = L

    def set_W(self, W):
        self.W = W

    def get_Gain(self, theta_phi):
        """
        Parameters: thetaPhi numpy array in the form of [theta,phi]
        Return: gain at given solid angle coordinates (local reference system)
        """
        #return self.idealGain(theta_phi)
        gain_track = 1j * np.zeros_like(theta_phi[0])
        # -> old interpolator (it has problems when phi has an odd number of samples)
        # gain_track = cubic_interpolation(theta_phi, self.theta_ax, self.phi_ax, self.gain_matrix, self.cubic,
        #                                  gain_track)
        # -> new interpolator
        gain_track = sphere_interp(theta_phi[0], theta_phi[1],
                                   self.theta_ax, self.phi_ax,
                                   self.gain_matrix, gain_track,
                                   self.cubic)
        return gain_track


    def get_normalized_gain(self, theta_phi: np.ndarray):
        """
        :param: theta_phi
        :return: gain normalized to 1
        """
        return self.get_Gain(theta_phi) / self.broadside_gain

    def get_gain_at_lcs_point(self, position: np.ndarray, normalize=False) -> complex:
        """
        returns the gain normalized to the maximum gain
        :param position: 3xN array contining x,y,z coordinates relative to the satellite LCS
        :param normalize: optional, if true the gain pattern is normalized to the peak gain provided in antenna
        :return: gain vector complex 1xN
        """
        # 1 convert relative position into spherical coordinates:
        theta_phi = cart2sph(position)[1:3]
        # 2 interpolate gain at those solid angles
        gain_track = 1j * np.zeros_like(theta_phi[0])

        # -> old interpolator (it has problems when phi has an odd number of samples)
        #gain_track = cubic_interpolation(theta_phi, self.theta_ax, self.phi_ax, self.gain_matrix, self.cubic,
        #                                 gain_track)
        # -> new interpolator
        gain_track = sphere_interp(theta_phi[0], theta_phi[1],
                                   self.theta_ax, self.phi_ax,
                                   self.gain_matrix, gain_track,
                                   self.cubic)
        # 3 normalize with respect to broadside gain
        if normalize:
            gain_track /= self.broadside_gain
        return gain_track
        # return self.get_fast_simple_gain_lcs_point(position)


    def get_fast_simple_gain_lcs_point(self, position: np.ndarray) -> complex:
        """
        wrapper for jit compiled simplified antenna weight, needs to be changed for the general case
        :param position: np array 3 rows (xs, ys, zs) LCS and N columns (evolution in simulation time)
        :return:
        """
        # # just for compilation #
        # pluto = self.get_fast_simple_gain_jit(self.lamda, self.L, self.W, position[:,0:2])
        # for actual run
        return self.get_fast_simple_gain_jit(self.lamda, self.L, self.W, position)

    @staticmethod
    @jit(nopython=True, parallel=True)
    def get_fast_simple_gain_jit(lamda, L, W, position: np.ndarray) -> complex:
        """
        compiled version of simple antenna gain weight for a point given in lcs with respect to the antenna
        :param lamda:
        :param L:
        :param W:
        :param position:
        :return:
        """
        gain = 1j * np.zeros(len(position[0, :]))
        print("computing gain weighting")
        theta_phi = cart2sph(position)
        for ii in (prange(len(position[0, :]))):
            gain[ii] = fastIdealGain(lamda, L, W, theta_phi[1:3, ii]) / \
                       (4 * np.pi * L * W / lamda ** 2)
        return gain
