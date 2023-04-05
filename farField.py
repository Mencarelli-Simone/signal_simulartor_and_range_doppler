# by Simone Mencarelli
# created on 28.01.2022

##################### DEPENDENCIES ####################

import numpy as np
from numba import jit, prange
from numpy import cos, exp, sin
from scipy import integrate


################### NUMBA FUNCTIONS ###################

@jit(nopython=True, parallel=True)
def jmesh_field_transform(Theta: np.ndarray, Phi: np.ndarray, l_mesh: np.ndarray, w_mesh: np.ndarray,
                          tanEField: np.ndarray, f: np.ndarray, tmp: np.ndarray, c: float, k) -> np.ndarray:
    """
    performs the near field fourier transform over a meshgrid. can be accelerated by numba
    :param Theta: Theta meshgrid
    :param Phi: Phi meshgrid
    :param f: input far field meshgrid, shall be complex
    :param tmp: preallocated complex array w_mesh size
    :param c: speed of light
    :param k: wave number
    :return: the far field over f
    """
    # far field meshgrid (defined outside)
    #f = 1j * np.zeros_like(Theta)
    # aperture coordinates meshgrid
    X = l_mesh
    Y = w_mesh
    integrand = 1j * np.zeros_like(X)
    rows = integrand.shape[0]
    columns = integrand.shape[1]

    # field fourier transforms
    # [Orfanidis - Electromagnetic Waves and Antennas - ch 18]
    for it, ip in np.ndindex(f.shape):

        kx = k * cos(Phi[it, ip]) * sin(Theta[it, ip])
        ky = k * sin(Phi[it, ip]) * sin(Theta[it, ip])
        integrand = tanEField * exp(1j * kx * X + 1j * ky * Y)

        # temporary array to hold the inner integral values
        #tmp = 1j * np.zeros(rows)
        # inner integral loop
        for ii in prange(rows):
            tmp[ii] = np.trapz(
                            integrand[ii, :],
                            l_mesh[ii, :].astype(np.float64)     # dx
                        )
        # outer integral
        f[it, ip] = np.trapz(
                        tmp,
                        w_mesh.T[0, :].astype(np.float64)       # dy
                        )

    return f



#################### PROBLEM GEOMETRY #################
#                            ^ y
#                            |
#           <--------------- L --------------->
#           _________________|_________________    +
#          |                 |                 |   |
#          |       aperture  |          ->x-pol|   |
# ---------|-----------------.-----------------|---W---> x
#          |                 |  z       ^y-pol |   |
#          |                 |          |      |   |
#          |_________________|_________________|   +
#                            |
#                            |

####################### CLASSES #######################

# class to find a far field given a rectangular aperture with arbitrary E-field excitation
class Aperture:

    def __init__(self, length: float, width: float, frequency: float = 10e9):
        """
        aperture mesh shape
        :param length: total length
        :param width: total width
        :param frequency: working frequency of the antenna
        """
        # antenna width along y
        self.W = width  # [m]
        # antenna length along x
        self.L = length  # [m]
        # working frequency
        self.freq = frequency  # [Hz]
        # light speed
        self.c = 299792458  # [m]
        # wave number
        self.k = 2 * np.pi * self.freq / self.c # note, k can be changed in case of a slant plane wave
        # input power for normalization of the intensity pattern
        self.inputPower = 1  # [W]
        # free space impedance
        self.eta = 376.7303136686

        # uninitialized parameters:
        self.l_mesh = None
        self.w_mesh = None
        self.tanEField = None

    def set_uniform_mesh(self, l_points: int, w_points: int):
        """
        generates the mesh for aperture sampling (regularly spaced samples)
        :param l_points: total number of points along L
        :param w_points: total number of points along W
        :return:
        """
        # odd numbers only
        if l_points % 2 == 0:
            l_points += 1
        if w_points % 2 == 0:
            w_points += 1

        self.l_mesh = np.linspace(-self.L / 2, self.L / 2, l_points)
        self.w_mesh = np.linspace(-self.W / 2, self.W / 2, w_points)
        self.tanEField = 1j * np.zeros((len(self.l_mesh), len(self.w_mesh))).T

    def set_uniform_mesh_resolution(self, delta_l: float, delta_w: float):
        """
        generates the mesh for aperture sampling (regularly spaced samples)
        :param delta_l: sampling resolution along L
        :param delta_w: sampling resolution along W
        :return:
        """
        l_points = np.ceil(self.L / delta_l).astype(int)
        w_points = np.ceil(self.W / delta_w).astype(int)
        self.set_uniform_mesh(l_points, w_points)

    def theor_rect_field_transform(self, theta_mesh: np.ndarray, phi_mesh: np.ndarray)-> np.ndarray:
        """
        generate and return the far field (aperture fourier transform) over a spherical coordinates grid
        only one linear polarization at a time can be computed. The sources is assumed to be
        ar rectangular uniform Huygens source
        :param theta_mesh: elevation points meshgrid
        :param phi_mesh: azimuth points meshgrid
        :return: matrix containing the theta-phi complex values of the far-field pattern
        """
        # meshgrid for aperture sampling
        l_mesh, w_mesh = np.meshgrid(self.l_mesh, self.w_mesh)
        kx = self.k * cos(phi_mesh) * sin(theta_mesh)
        ky = self.k * sin(phi_mesh) * sin(theta_mesh)
        f = sin(kx * self.L / 2) / (kx * self.L / 2) * sin(ky * self.W / 2) / (ky * self.W / 2) * self.L * self.W
        return theta_mesh, phi_mesh, f

    def field_transform(self, theta: np.ndarray, phi: np.ndarray, interpolation="simpson", polarization= "y" ) -> np.ndarray:
        """
        generate and return the far field (aperture fourier transform) over a spherical coordinates grid
        only one linear polarization at a time can be computed. The sources is assumed to be
        an Huygens source
        :param theta: elevation points
        :param phi: azimuth points
        :param interpolation: "simpson" or "trapezoidal", interpolation technique to be used
        :param polarization: "x" or "y" e field orientation axis
        :return: matrix containing the theta-phi complex values of the far-field pattern
        """
        # field samples initialization
        f = 1j * np.zeros((len(theta), len(phi)))
        # sph. coordinates meshgrid
        Theta, Phi = np.meshgrid(theta, phi)
        # aperture coordinates meshgrid
        X, Y = np.meshgrid(self.l_mesh, self.w_mesh)
        integrand = 1j * np.zeros_like(X)

        # field fourier transforms
        # [Orfanidis - Electromagnetic Waves and Antennas - ch 18]
        for it, ip in np.ndindex(f.shape):
            kx = self.k * cos(Phi[ip, it]) * sin(Theta[ip, it])
            ky = self.k * sin(Phi[ip, it]) * sin(Theta[ip, it])
            integrand = self.tanEField * exp(1j * kx * X + 1j * ky * Y)
            if interpolation == "simpson":
                # fourier transform using simpson method
                f[it, ip] = \
                    integrate.simps(
                        integrate.simps(
                            integrand,
                            self.l_mesh,  # dx
                            axis=1
                        ),
                        self.w_mesh  # dy
                    )
            else:
                # fourier transform using trapezoidal integration method
                f[it, ip] = \
                    integrate.trapz(
                        integrate.trapz(
                            integrand,
                            self.l_mesh,  # dx
                            axis=1
                        ),
                        self.w_mesh  # dy
                    )
        # Theta meshgrid
        self.Theta = Theta
        # Phi meshgrid
        self.Phi = Phi
        self.f = f
        return Theta, Phi, f

    def mesh_field_transform(self, theta_mesh: np.ndarray, phi_mesh: np.ndarray, polarization= "y") -> np.ndarray:
        """
        performs the far field transform over a 2-d meshgrid
        :param theta_mesh: theta points meshgrid
        :param phi_mesh: phi points meshgrid
        :param polarization: optional x or y
        :return: far field
        """
        # initialize the far field vector
        f = 1j * np.zeros_like(theta_mesh)
        # meshgrid for aperture sampling
        l_mesh, w_mesh = np.meshgrid(self.l_mesh, self.w_mesh)
        tmp = 1j * np.zeros_like(self.w_mesh)
        f = jmesh_field_transform(theta_mesh,
                                  phi_mesh,
                                  l_mesh,
                                  w_mesh,
                                  self.tanEField,
                                  f,
                                  tmp,
                                  self.c,
                                  self.k)
        self.f = f
        return theta_mesh, phi_mesh, f



# uniform aperture (extended from Aperture)
class UniformAperture(Aperture):
    def __init__(self, length: float, width: float, frequency: float = 10e9):
        Aperture.__init__(self, length, width, frequency)

        # setting the uniform field on the aperture
        self.e_amplitude = 1  # [v/m]
        self.set_uniform_mesh_resolution(.1 * self.c / self.freq, .1 * self.c / self.freq)
        self.tanEField += self.e_amplitude * np.ones_like(self.tanEField)

    def gain_pattern(self, theta: np.ndarray, phi: np.ndarray, interpolation="simpson", polarization = "y"):
        """
        calculate the gain for a hiuygenes source
        :param theta: elevation points
        :param phi: azimuth points
        :param interpolation: "simpson" or "trapezoidal", interpolation technique to be used
        :param polarization: "x" or "y" e field orientation axis
        :return: matrix containing the theta-phi complex values of the far-field pattern
        """
        self.Theta, self.Phi, self.f = self.field_transform(theta, phi, interpolation, polarization)
        # obliquity factors for phased apertures need to be changed
        c_t = 1/2 * (np.ones_like(self.Theta) + cos(self.Theta))
        c_p = 1/2 * (np.ones_like(self.Theta) + cos(self.Theta))
        if polarization == "x":
            u = self.k**2 / (8 * np.pi**2 * self.eta) * (c_t.T**2 * self.f**2)
        if polarization == "y":
            u = self.k**2 / (8 * np.pi**2 * self.eta) * (c_t.T**2 * self.f**2)
            # no difference in this case
        else:
            print("Error, polarization shall be either x or y")
        self.g = np.abs(u) * self.eta * 8 * np.pi / (self.W*self.L)
        return self.g

    def mesh_gain_pattern(self, theta_mesh: np.ndarray, phi_mesh: np.ndarray, polarization = "y"):
        """
        calculate the gain for a hiuygenes source
        :param theta: elevation points
        :param phi: azimuth points
        :param interpolation: "simpson" or "trapezoidal", interpolation technique to be used
        :param polarization: "x" or "y" e field orientation axis
        :return: matrix containing the theta-phi complex values of the far-field pattern
        """
        self.Theta, self.Phi, self.f = self.mesh_field_transform(theta_mesh, phi_mesh, polarization)
        # obliquity factors for phased apertures need to be changed
        c_t = 1/2 * (np.ones_like(self.Theta) + cos(self.Theta))
        c_p = 1/2 * (np.ones_like(self.Theta) + cos(self.Theta))
        # c_phi and c_theta are different for modified hiuygenes sources
        if polarization == "x":
            u = (self.k**2 / (8 * np.pi**2 * self.eta)) * (c_t**2 * self.f**2)
        if polarization == "y":
            u = (self.k**2 / (8 * np.pi**2 * self.eta)) * (c_t**2 * self.f**2)
            # no difference in this case: simplify eq 18.6.3 to single pol non steered
        else:
            print("Error, polarization shall be either x or y")

        # gain from range normalized radiance
        self.g = np.abs(u) * self.eta * 8 * np.pi / (self.W * self.L)

        return self.g

    def mesh_gain_pattern_theor(self, theta_mesh: np.ndarray, phi_mesh: np.ndarray, polarization = "y"):
        """
        calculate the gain for a hiuygenes source
        :param theta: elevation points
        :param phi: azimuth points
        :param interpolation: "simpson" or "trapezoidal", interpolation technique to be used
        :param polarization: "x" or "y" e field orientation axis
        :return: matrix containing the theta-phi complex values of the far-field pattern
        """
        self.Theta, self.Phi, self.f = self.theor_rect_field_transform(theta_mesh, phi_mesh)
        # obliquity factors for phased apertures need to be changed
        c_t = 1/2 * (np.ones_like(self.Theta) + cos(self.Theta))
        c_p = 1/2 * (np.ones_like(self.Theta) + cos(self.Theta))
        # c_phi and c_theta are different for modified hiuygenes sources
        if polarization == "x":
            u = (self.k**2 / (8 * np.pi**2 * self.eta)) * (c_t**2 * self.f**2)
        if polarization == "y":
            u = (self.k**2 / (8 * np.pi**2 * self.eta)) * (c_t**2 * self.f**2)
            # no difference in this case: simplify eq 18.6.3 to single pol non steered
        else:
            print("Error, polarization shall be either x or y")

        # gain from range normalized radiance
        g = np.abs(u) * self.eta * 8 * np.pi / (self.W * self.L)

        return g

    def max_gain(self):
        """
        :return: the peak gain of pattern
        """
        t = np.zeros((1,1))
        p = np.zeros((1,1))
        t, p, A = self.field_transform(t,p)
        A = np.abs(A.sum())
        max_g = np.pi * 4 * A / (self.c / self.freq) ** 2
        return max_g


### TEST ####
if __name__ == "__main__":
    uniap = UniformAperture(4,.8)
    theta = np.linspace(0, np.pi / 2, 100)
    phi = np.linspace(0, np.pi / 2, 50)
    # method 1
    #g1 = uniap.gain_pattern(theta, phi)# interpolation= "trapz")
    # method 2
    Theta, Phi = np.meshgrid(theta, phi)
    g2 = uniap.mesh_gain_pattern(Theta, Phi) # it underestimates, but just a bit

    # theoretical
    gt = uniap.mesh_gain_pattern_theor(Theta, Phi)
    # plot
    import matplotlib.pyplot as plt

    # fig, ax  = plt.subplots(1)
    # c = ax.pcolormesh(uniap.Theta * cos(uniap.Phi), uniap.Theta * sin(uniap.Phi) ,10*np.log10(np.abs(g1.T)), cmap=plt.get_cmap('hot'))
    # fig.colorbar(c)
    # ax.set_xlabel("$\\theta\  cos \phi$")
    # ax.set_ylabel("$\\theta\  sin \phi$")

    fig, ax  = plt.subplots(1)
    c = ax.pcolormesh(uniap.Theta * cos(uniap.Phi), uniap.Theta * sin(uniap.Phi) ,10*np.log10(g2), cmap=plt.get_cmap('hot'))
    fig.colorbar(c)
    ax.set_xlabel("$\\theta\  cos \phi$")
    ax.set_ylabel("$\\theta\  sin \phi$")
    # THEORETICAL
    fig, ax = plt.subplots(1)
    c = ax.pcolormesh(uniap.Theta * cos(uniap.Phi), uniap.Theta * sin(uniap.Phi), 10 * np.log10(gt),
                      cmap=plt.get_cmap('hot'))
    fig.colorbar(c)
    ax.set_xlabel("$\\theta\  cos \phi$")
    ax.set_ylabel("$\\theta\  sin \phi$")
    # DIFFERENCE
    fig, ax = plt.subplots(1)
    c = ax.pcolormesh(uniap.Theta * cos(uniap.Phi), uniap.Theta * sin(uniap.Phi), ((g2-gt)/g2),
                      cmap=plt.get_cmap('hot'))
    fig.colorbar(c)
    ax.set_xlabel("$\\theta\  cos \phi$")
    ax.set_ylabel("$\\theta\  sin \phi$")
  #  ax.set_xlim(0, np.pi/4)
  #  ax.set_ylim(0, np.pi/4)

  # todo cuts