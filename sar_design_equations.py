# contains some equations to be used in the radar design


from geometryRadar import RadarGeometry
import numpy as np


def adjust_looking_angle_to_prf(radar_geometry: RadarGeometry, prf, c=299792458, fraction=1 / 2):
    """
    Centers the returns from the broadside direction on ground to half PRI exactly or a desired fraction of PRI
    :param radar_geometry: RadarGeometry object containing the position and rotation of the antenna
    :param c: optional, wave propagation speed, default 299792458 m/s
    :param fraction: optional, position of the broadside on ground within a pulse interval
    :return: new side looking angle
    """
    # find broadside on ground
    bsg = radar_geometry.get_broadside_on_ground().reshape((3, 1))
    # find range
    r0 = np.linalg.norm(radar_geometry.S_0 - bsg)
    # find how many pulses it takes for a return to bounce back at that range
    n_pulses = int(2 * r0 / c * prf)
    # desired range
    r0_new = (n_pulses + fraction) * c / (2 * prf)
    print('old range ', r0, ', new range ', r0_new)
    # the new looking angle is given by the following formula
    cos_beta = radar_geometry.S_0[2, 0] / (r0_new * np.cos(radar_geometry.forward_squint_angle))
    beta = np.arccos(cos_beta)
    print('old gazing angle deg ', radar_geometry.side_looking_angle * 180 / np.pi, ' new gazing angle deg ',
          beta * 180 / np.pi)
    # setting in the new side loking angle
    radar_geometry.set_rotation(beta, radar_geometry.trajectory_angle, radar_geometry.forward_squint_angle)
    return beta


def adjust_prf_to_looking_angle(radar_geometry: RadarGeometry, prf, c=299792458, fraction=1 / 2):
    """
    Centers the returns from the broadside direction on ground to half PRI exactly or a desired fraction of PRI
    by slightly increasing the prf
    :param radar_geometry: RadarGeometry object containing the position and rotation of the antenna
    :param prf: original prf, pulse repetition frequency
    :param c: optional, wave propagation speed, default 299792458 m/s
    :param fraction: optional, position of the broadside on ground within a pulse interval
    :return: new prf
    """
    # find broadside on ground
    bsg = radar_geometry.get_broadside_on_ground().reshape((3, 1))
    # find range
    r0 = np.linalg.norm(radar_geometry.S_0 - bsg)
    # find how many pulses it takes for a return to bounce back at that range
    n_pulses = int(2 * r0 / c * prf)
    # desired new prf
    prf_new = c * (n_pulses + fraction) / (2 * r0)
    print('old PRF ', prf, ', new PRF ', prf_new)
    return prf_new


def prf_to_ground_swath(radar_geometry: RadarGeometry, prf, c=299792458):
    """
    return the minimum and maximum ground range points given by the prf
    :param radar_geometry: RadarGeometry object containing the position and rotation of the antenna
    :param prf: original prf, pulse repetition frequency
    :param c: optional, wave propagation speed, default 299792458 m/s
    :return:
    """
    # find broadside on ground
    bsg = radar_geometry.get_broadside_on_ground().reshape((3, 1))
    # find range
    r0 = np.linalg.norm(radar_geometry.S_0 - bsg)
    # find how many pulses it takes for a return to bounce back at that range
    n_pulses = int(2 * r0 / c * prf)
    # minimum true range
    r_min = n_pulses * c / (2 * prf)
    # maximum true range
    r_max = (1 + n_pulses) * c / (2 * prf)
    # converting r_min to ground range min
    r1_min = r_min * np.cos(radar_geometry.forward_squint_angle)
    rg_min = np.sqrt(r1_min**2 - radar_geometry.S_0[2, 0]**2)
    # converting r_max to ground range max
    r1_max = r_max * np.cos(radar_geometry.forward_squint_angle)
    rg_max = np.sqrt(r1_max**2 - radar_geometry.S_0[2, 0]**2)
    swath = rg_max - rg_min
    print('swath on ground: ', swath,'[m] ground range from ', rg_min,'[m] to ', rg_max,'[m]')
    return swath , rg_min, rg_max


## test
if __name__ == '__main__':
    # setup a radar geometry
    geome = RadarGeometry()
    beta = 40
    geome.set_rotation(beta * np.pi / 180, 0, 0)
    radar_altitude = 500E3  # m
    geome.set_initial_position(0, 0, radar_altitude)
    geome.set_speed(geome.orbital_speed())
    # adjust the looking angle
    prf = 15224.562293834051
    # adjust_looking_angle_to_prf(geome, prf)
    adjust_prf_to_looking_angle(geome, prf)
    prf_to_ground_swath(geome, prf)
