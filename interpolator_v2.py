import matplotlib.pyplot as plt
import numpy as np
from numba import prange, jit



@jit(nopython=True, parallel=True)
def sphere_interp(theta_out, phi_out, theta_ax, phi_ax, pattern, out_pattern, cubic: bool = True):
    """

    :param theta_out: array_like 1d theta output points
    :param phi_out: array_like 1d phi output points
    :param theta_ax: ordered homogeneously sampled theta axis
    :param phi_ax: ordered homogeneously sampled theta axis
    :param pattern: data to interpolate 2-d matrix
    :param cubic: optional, set to false to use rectangular
    :return: interpolated pattern array_like 1d as theta and phi
    """
    # output array
    # out_pattern = np.zeros_like(theta_out).astype(np.complex128)
    # explicit casting
    #pattern = pattern.astype(np.complex128)

    # find the min max step of axes
    theta_min = theta_ax[0]
    theta_max = theta_ax[-1]
    theta_step = (theta_ax[-1] - theta_ax[0]) / (len(theta_ax)-1)

    # unwrapping phi
    phi_ax = np.where(phi_ax < 0, np.pi * 2 + phi_ax, phi_ax)
    phi_min = phi_ax[0]
    phi_max = phi_ax[-1]
    phi_step = (phi_ax[-1] - phi_ax[0]) / (len(phi_ax) - 1)

    phi_out = np.where(phi_out < 0, np.pi * 2 + phi_out, phi_out)
    # find 0 1 2 3 indices
    print('finding indices')
    theta_idx_0 = np.floor(((theta_out - theta_step) % (2 * np.pi) - theta_min) / theta_step).astype(np.int64)
    theta_idx_1 = np.floor((theta_out % (2 * np.pi) - theta_min) / theta_step).astype(np.int64)
    theta_idx_2 = np.floor(((theta_out + theta_step) % (2 * np.pi) - theta_min) / theta_step).astype(np.int64)
    theta_idx_3 = np.floor(((theta_out + 2 * theta_step) % (2 * np.pi) - theta_min) / theta_step).astype(np.int64)

    theta_idx_0 = np.where(theta_idx_0 >= int(np.pi / theta_step), -theta_idx_0, theta_idx_0)
    theta_idx_1 = np.where(theta_idx_1 >= int(np.pi / theta_step), -theta_idx_1, theta_idx_1)
    theta_idx_2 = np.where(theta_idx_2 >= int(np.pi / theta_step), -theta_idx_2, theta_idx_2)
    theta_idx_3 = np.where(theta_idx_3 >= int(np.pi / theta_step), -theta_idx_3, theta_idx_3)

    phi_idx_0 = np.floor(((phi_out - phi_step) % (2 * np.pi) - phi_min) / phi_step).astype(np.int64)
    phi_idx_1 = np.floor((phi_out % (2 * np.pi) - phi_min) / phi_step).astype(np.int64)
    phi_idx_2 = np.floor(((phi_out + phi_step) % (2 * np.pi) - phi_min) / phi_step).astype(np.int64)
    phi_idx_3 = np.floor(((phi_out + 2 * phi_step) % (2 * np.pi) - phi_min) / phi_step).astype(np.int64)

    print('checking indices')
    # check edges and return eventual errors
    maxidx = max(theta_idx_0.max(), theta_idx_1.max(), theta_idx_2.max(), theta_idx_3.max())
    minidx = min(theta_idx_0.min(), theta_idx_1.min(), theta_idx_2.min(), theta_idx_3.min())
    if maxidx >= len(theta_ax) + 2:
        print(" Sphere interpolation error, theta index outside boundaries")
        print(maxidx, minidx, theta_min, theta_max, len(theta_ax), theta_step)
        return
    else:
        # retain index for edge values
        theta_idx_0 = np.where(theta_idx_0 > len(theta_ax) - 1, len(theta_ax) - 1, theta_idx_0)
        theta_idx_1 = np.where(theta_idx_1 > len(theta_ax) - 1, len(theta_ax) - 1, theta_idx_1)
        theta_idx_2 = np.where(theta_idx_2 > len(theta_ax) - 1, len(theta_ax) - 1, theta_idx_2)
        theta_idx_3 = np.where(theta_idx_3 > len(theta_ax) - 1, len(theta_ax) - 1, theta_idx_3)

        theta_idx_0 = np.where(theta_idx_0 < 0, 0, theta_idx_0)
        theta_idx_1 = np.where(theta_idx_1 < 0, 0, theta_idx_1)
        theta_idx_2 = np.where(theta_idx_2 < 0, 0, theta_idx_2)
        theta_idx_3 = np.where(theta_idx_3 < 0, 0, theta_idx_3)

    maxidx = max(phi_idx_0.max(), phi_idx_1.max(), phi_idx_2.max(), phi_idx_3.max())
    minidx = min(phi_idx_0.min(), phi_idx_1.min(), phi_idx_2.max(), phi_idx_3.min())
    if maxidx >= len(phi_ax) + 2 or minidx < 0:
        print(" Sphere interpolation error, phi index outside boundaries")
        print(maxidx, minidx, phi_min, phi_max, len(phi_ax), phi_step)
        return
    else:
        # retain index for edge values
        phi_idx_0 = np.where(phi_idx_0 > len(phi_ax) - 1, len(phi_ax) - 1, phi_idx_0)
        phi_idx_1 = np.where(phi_idx_1 > len(phi_ax) - 1, len(phi_ax) - 1, phi_idx_1)
        phi_idx_2 = np.where(phi_idx_2 > len(phi_ax) - 1, len(phi_ax) - 1, phi_idx_2)
        phi_idx_3 = np.where(phi_idx_3 > len(phi_ax) - 1, len(phi_ax) - 1, phi_idx_3)

        phi_idx_0 = np.where(phi_idx_0 < 0, 0, phi_idx_0)
        phi_idx_1 = np.where(phi_idx_1 < 0, 0, phi_idx_1)
        phi_idx_2 = np.where(phi_idx_2 < 0, 0, phi_idx_2)
        phi_idx_3 = np.where(phi_idx_3 < 0, 0, phi_idx_3)

    print('interpolating')

    if cubic:
        # parallel interpolation
        for ii in prange(len(out_pattern)):
            p = np.zeros((4)).astype(np.complex128)
            temp = np.zeros((4)).astype(np.complex128)
            # 1 -  we interpolate along phi to find four theta points
            # output theta displacement within cell
            x_t = (theta_out[ii] - theta_idx_1[ii] * theta_step) / theta_step
            phi_idxs = np.zeros(4).astype(np.int64)
            phi_idxs[0] = phi_idx_0[ii]
            phi_idxs[1] = phi_idx_1[ii]
            phi_idxs[2] = phi_idx_2[ii]
            phi_idxs[3] = phi_idx_3[ii]
            # interpolating the 4 points along the output theta coordinate
            for jj in range(4):
                # four points of the interpolation
                p[0] = pattern[theta_idx_0[ii], phi_idxs[jj]]
                p[1] = pattern[theta_idx_1[ii], phi_idxs[jj]]
                p[2] = pattern[theta_idx_2[ii], phi_idxs[jj]]
                p[3] = pattern[theta_idx_3[ii], phi_idxs[jj]]
                # inner interpolation
                temp[jj] = p[1] + 0.5 * x_t * (p[2] - p[0] + x_t * (
                            2.0 * p[0] - 5.0 * p[1] + 4.0 * p[2] - p[3] + x_t * (
                            3.0 * (p[1] - p[2]) + p[3] - p[0])))
            # 2 -  finding the final point interpolating temp along phi
            # output phi displacement within cell
            x_p = (phi_out[ii] - phi_idx_1[ii] * phi_step) / phi_step
            point = temp[1] + 0.5 * x_p * (temp[2] - temp[0] + x_p * (
                            2.0 * temp[0] - 5.0 * temp[1] + 4.0 * temp[2] - temp[3] + x_p * (
                            3.0 * (temp[1] - temp[2]) + temp[3] - temp[0])))
            out_pattern[ii] = point

    else:
        # parallel rect interpolation

        for ii in prange(len(out_pattern)):
            temp = np.zeros(2).astype(np.complex128)
            p = np.zeros(2).astype(np.complex128)
            # 1 -  we interpolate along phi to find 2 theta points
            # output theta displacement within cell
            x_t = (theta_out[ii] - theta_idx_1[ii] * theta_step) / theta_step
            phi_idxs = np.zeros(2).astype(np.int64)
            phi_idxs[0] = phi_idx_1[ii]
            phi_idxs[1] = phi_idx_2[ii]

            # interpolating the 4 points along the output theta coordinate
            for jj in range(2):
                # four points of the interpolation
                p[0] = pattern[theta_idx_1[ii], phi_idxs[jj]]
                p[1] = pattern[theta_idx_2[ii], phi_idxs[jj]]
                temp[jj] = p[0] + x_t * (p[1]- p[0])


            # 2 -  finding the final point interpolating temp along phi
            # output phi displacement within cell
            x_p = (phi_out[ii] - phi_idx_1[ii] * phi_step) / phi_step
            out_pattern[ii] = temp[0] + x_p * (temp[1]-temp[0])

    print('returning')
    return out_pattern

if __name__ == '__main__':
    from antenna import Antenna
    # this will load the pattern we already have
    antenna = Antenna(path='./Antenna_Pattern')

    theta = np.linspace(0, np.pi /32 , 1500)
    phi = np.linspace(0, np.pi * 2 , 1200)
    Th, Ph = np.meshgrid(theta, phi)
    pattern_interp = 1j * np.ones_like(np.ravel(Th))
    pattern_interp = sphere_interp(np.ravel(Th.T), np.ravel(Ph.T), antenna.theta_ax, antenna.phi_ax, antenna.gain_matrix, pattern_interp, cubic=True)
    pattern_interp = pattern_interp.reshape(Th.T.shape)

    # %% display
    fig, ax = plt.subplots(1)
    ax.imshow(10*np.log10(np.abs(antenna.gain_matrix)))

    fig, ax = plt.subplots(1)
    ax.imshow(10*np.log10(np.abs(pattern_interp)))

    #%%
    fig, ax  = plt.subplots(1)
    c = ax.pcolormesh(Th * np.cos(Ph), Th * np.sin(Ph) ,10*np.log10(np.abs(pattern_interp.T)), cmap=plt.get_cmap('hot'))
    #c = ax.pcolormesh(Phi * cos(Theta), Phi * sin(Theta) ,10*np.log10(np.abs(pat.gain_pattern)), cmap=plt.get_cmap('hot'))
    fig.colorbar(c)
    ax.set_xlabel("$\\theta\  cos \phi$")
    ax.set_ylabel("$\\theta\  sin \phi$")