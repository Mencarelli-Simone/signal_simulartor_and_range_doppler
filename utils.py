##  ____________________________imports_____________________________
import numpy as np

## ____________________________functions_____________________________
# polar to rectangular
from numba import jit, prange


@jit(nopython = True)
def sph2cart(P: np.ndarray) -> np.ndarray:
    """ Convert a numpy array in the form [r,theta,phi] to a numpy array in the form [x,y,z]
  """
    r, theta, phi = 0, 1, 2
    x = P[r] * np.sin(P[theta]) * np.cos(P[phi])
    y = P[r] * np.sin(P[theta]) * np.sin(P[phi])
    z = P[r] * np.cos(P[theta])
    return np.array((x, y, z))


# rectangular to polar
@jit(nopython = True)
def cart2sph(P:np.ndarray) -> np.ndarray:
    """ Convert a numpy array in the form [x,y,z] to a numpy array in the form [r,theta,phi]
    """
    x, y, z = 0, 1, 2
    #r = np.linalg.norm(P)
    r = np.sqrt(P[x]**2 + P[y]**2 + P[z]**2).astype(np.float64)
    theta = np.where(r != 0, np.arccos(P[z] / r), 0).astype(np.float64)
    phi = np.where(r != 0, np.arctan2(P[y], P[x]), 0).astype(np.float64)
    return np.stack((r, theta, phi))

# polar to rectangular meshgrid
@jit(nopython=True, parallel=True)
def meshSph2cart(r_mesh, theta_mesh, phi_mesh):
    """

    :param r_mesh: r coordinates meshgrid
    :param theta_mesh: theta coordinates meshgrid
    :param phi_mesh: phi coordinates meshgrid
    :return: x, y, z in gcs
    """
    x = np.zeros_like(r_mesh).astype(np.float64)
    y = np.zeros_like(theta_mesh).astype(np.float64)
    z = np.zeros_like(phi_mesh).astype(np.float64)

    rows, columns = r_mesh.shape
    for rr in prange(rows):
        for cc in prange(columns):
            x = r_mesh[rr, cc] * np.sin(theta_mesh[rr, cc]) * np.cos(phi_mesh[rr, cc])
            y = r_mesh[rr, cc] * np.sin(theta_mesh[rr, cc]) * np.sin(phi_mesh[rr, cc])
            z = r_mesh[rr, cc] * np.cos(theta_mesh[rr, cc])

    return x, y, z

# Rectangular to spherical meshgrid
@jit(nopython=True, parallel=True)
def meshCart2sph(x_mesh, y_mesh, z_mesh):
    """
    from rec to sph
    :param x_mesh: x coordinates meshgrid
    :param y_mesh: y coordinates meshgrid
    :param z_mesh: z coordinates meshgrid
    :return: r_mesh, theta_mesh, phi_mesh
    """
    r_mesh = np.zeros_like(x_mesh).astype(np.float64)
    theta_mesh = np.zeros_like(x_mesh).astype(np.float64)
    phi_mesh = np.zeros_like(x_mesh).astype(np.float64)

    rows, columns = x_mesh.shape
    for rr in prange(rows):
        for cc in prange(columns):
            r_mesh[rr, cc] = np.sqrt(x_mesh[rr, cc] ** 2 + y_mesh[rr, cc] ** 2 + z_mesh[rr, cc] ** 2)
            theta_mesh[rr, cc] = np.where(r_mesh[rr, cc] != 0, np.arccos(z_mesh[rr, cc] / r_mesh[rr, cc]), 0.0)
            phi_mesh[rr, cc] = np.where(r_mesh[rr, cc] != 0, np.arctan2(y_mesh[rr, cc], x_mesh[rr, cc]), 0.0)

    return r_mesh, theta_mesh, phi_mesh


# fast implementation for change of coordinates
@jit(nopython=True, parallel=True)
def mesh_lcs_to_gcs(x_mesh, y_mesh, z_mesh, Bs2c, S0):
    """

    :param x_mesh: lcs x coordinates meshgrid
    :param y_mesh: lcs y coordinates meshgrid
    :param z_mesh: lcs z coordinates meshgrid
    :param Bs2c: matrix of basis change from local to global
    :param S0: position of the local coordinate system
    :return: x, y, z in gcs
    """
    x = np.zeros_like(x_mesh).astype(np.float64)
    y = np.zeros_like(y_mesh).astype(np.float64)
    z = np.zeros_like(z_mesh).astype(np.float64)

    rows, columns = x_mesh.shape
    for rr in prange(rows):
        for cc in prange(columns):
            P = S0 + Bs2c @ np.array([[x_mesh[rr, cc]], [y_mesh[rr, cc]], [z_mesh[rr, cc]]])
            x[rr, cc], y[rr, cc], z[rr, cc] = P[0], P[1], P[2]

    return x, y, z


# fast implementation for change of coordinates
@jit(nopython=True, parallel=True)
def mesh_gcs_to_lcs(x_mesh, y_mesh, z_mesh, Bc2s, S0):
    """

    :param x_mesh: gcs x coordinates meshgrid
    :param y_mesh: gcs y coordinates meshgrid
    :param z_mesh: gcs z coordinates meshgrid
    :param Bc2s: matrix of basis change from global to local
    :param S0: position of the local coordinate system
    :return: x, y, z meshgrids in lcs
    """
    x = np.zeros_like(x_mesh).astype(np.float64)
    y = np.zeros_like(y_mesh).astype(np.float64)
    z = np.zeros_like(z_mesh).astype(np.float64)

    rows, columns = x_mesh.shape
    for rr in prange(rows):
        for cc in prange(columns):
            P = Bc2s @ (np.array([[x_mesh[rr, cc]], [y_mesh[rr, cc]], [z_mesh[rr, cc]]]) - S0)
            x[rr, cc] = P[0, 0]
            y[rr, cc] = P[1, 0]
            z[rr, cc] = P[2, 0]

    return x, y, z


# Stationary phase points in posp approximation for azimuth-doppler conversion
@jit(nopython=True, parallel=True)
def mesh_doppler_to_azimuth(range_mesh, doppler_mesh, lambda_c, v_s):
    """
    converts range-doppler coordinates to range-azimuth coordinates using the stationary phase approximation for the
    azimuth chirp of the radar.
    :param range_mesh: range coordinates meshgrid
    :param doppler_mesh: doppler coordinates meshgrid
    :param lambda_c: signal modulation wavelength
    :param v_s: speed of the platform
    :return: range_mesh, azimuth_mesh new coordinates
    """
    rows, columns = range_mesh.shape
    slow_time = np.zeros_like(range_mesh)
    for rr in prange(rows):
        for cc in prange(columns):
            slow_time[rr, cc] = - doppler_mesh[rr, cc] * lambda_c * range_mesh[rr, cc] / \
                        (v_s * np.sqrt(4 * v_s**2 - doppler_mesh[rr, cc]**2 * lambda_c**2))
    # transforming the slow_time into azimuth, by multiplying for the satellite speed
    slow_time *= v_s # slow time is actually azimuth now, I'm not creating a new array to save memory
    return range_mesh, slow_time


@jit(nopython=True, parallel=True)
def mesh_azimuth_range_to_ground_gcs(range_mesh, azimuth_mesh, velocity, S_0):
    """
    converts azimuth range points to gcs points on ground
    :param range_mesh: range coordinates meshgrid
    :param azimuth_mesh: azimuth coordinates meshgrid
    :param velocity: radar velocity versor ( it has to be parallel to ground)
    :param S_0: radar position at t = 0
    :return: x_mesh, y_mesh gcs ground coordinates
    """
    # shaping
    velocity = velocity.reshape((3,1))
    # 2d conversion and normalization
    v = velocity[0:2] / np.linalg.norm(velocity)
    # across track ground versor
    v_ortho = np.array([[0, 1.0], [-1.0, 0]]) @ v  # orthogonal versor

    # mesh iteration
    rows, columns = range_mesh.shape
    x_mesh = np.zeros_like(range_mesh)
    y_mesh = np.zeros_like(range_mesh)
    for rr in prange(rows):
        for cc in prange(columns):
            point = azimuth_mesh[rr, cc] * v + S_0[0:2]  # azimuth shift
            point += v_ortho * np.sqrt(range_mesh[rr, cc] ** 2 - S_0[2] ** 2)  # ground range shift
            x_mesh[rr, cc] = point[0, 0]
            y_mesh[rr, cc] = point[1, 0]

    return x_mesh, y_mesh