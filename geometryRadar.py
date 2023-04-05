#  ____________________________imports_____________________________
import numpy as np
from numpy import cos, sin



# ___________________________________________ Classes ______________________________________________

class RadarGeometry():
    """
    describes the rotation, speed and position of the radar with respect to a global reference system
    provides methods to retrieve radar position in time and taget related delays
    """
    def set_rotation(self, looking_angle:float, trajectory_angle:float, squint_angle:float):
        """
        sets the rotation of the satellite antenna and creates a reference system with axes accordingly rotated
        this sets also the velocity accordingly to. all angles are in radiant
        :param looking_angle: side looking angle
        :param trajectory_angle: trajectory angle on GCS xy plane
        :param squint_angle: forward looking squint angle
        :return: lcs
        """
        # rotation around z(alpha)`and then new around x(beta)  and then around y(gamma)
        self.trajectory_angle = trajectory_angle    # store it to be used in the speed vector
        self.side_looking_angle = looking_angle     # looking angle from the down looking direction to the right of the azimuth path
        self.forward_squint_angle = squint_angle    # forward looking angle toward the walk path
        alpha = trajectory_angle
        beta = looking_angle
        gamma = -squint_angle
        self.x_s = np.array([
            cos(alpha) * cos(gamma) + sin(alpha) * sin(beta) * sin(gamma),
            sin(alpha) * cos(gamma) - cos(alpha) * sin(beta) * sin(gamma),
            -cos(beta) * sin(gamma)
        ])
        self.y_s = np.array([
            sin(alpha) * cos(beta),
            -cos(alpha) * cos(beta),
            sin(beta)
        ])
        self.z_s = np.array([
            -cos(alpha) * sin(gamma) + sin(alpha) * sin(beta) * cos(gamma),
            -sin(alpha) * sin(gamma) - cos(alpha) * sin(beta) * cos(gamma),
            -cos(beta) * cos(gamma)
        ])
        # local basis matrix (transformation matrix between local and canonical basis)
        self.Bs2c = np.column_stack((self.x_s, self.y_s, self.z_s))
        # and vice versa
        self.Bc2s = self.Bs2c.T

        # update velocity vector
        if hasattr(self, 'abs_v'):
            self.set_speed(self.abs_v)

        return self.x_s, self.y_s, self.z_s

    def set_speed(self, speed:float):
        """
        sets the velocity vector of the radar(assuming in first approximation that the velocity is constant
        :param speed:
        :param trajectory_angle:
        :return:
        """
        alpha = self.trajectory_angle
        self.abs_v = speed
        self.velocity = self.abs_v * np.array((np.cos(alpha), np.sin(alpha), 0))
        self.velocity.shape = (3,1) # column vector
        return self.velocity

    def set_initial_position(self, x:float, y:float, z:float):
        """
        initial position with respect to the global reference system
        :param x: x GCS coordinate
        :param y: y GCS coordinate
        :param z: altitude i.e. z GCS coordinate
        :return: array form of initial position
        """
        self.S_0 = np.array([[x], [y], [z]]) # column
        return self.S_0

    def get_position(self, t):
        """
        get radar position in time
        this method can be overloaded to add imprecisions in the trajectory
        :param t:
        :return:
        """
        position = (np.ones_like(t) * self.S_0) + (t * self.velocity)
        return position

    def get_lcs_of_point(self, point_target, t):
        """
        get the local coordinates of a target in time as seen by the radar antenna
        :param point_target: numpy array 3-d containing the target gc
        :param t: time axis
        :return: 3 by len(t) lcs array
        """
        target = point_target
        # transpose target vector
        target.shape = (3, 1)
        # gcs range vector
        r = np.ones_like(t) * target - self.get_position(t) # todo, calculate position only once to improve multitarget
        # change to local
        return np.matmul(self.Bc2s, r)


    def get_range(self, point_target, t):
        """
        gets relative range between satellite and target in time
        :param point_target: numpy array of dimention 3 containing the gcs coordinates of the point target
        :param t:
        :return:
        """
        target = point_target
        target.shape = (3,1)
        # todo optimize this for multi-targets so that the position is computed once
        r = self.get_position(t) - target * np.ones_like(t)
        r = np.linalg.norm(r,axis = 0)
        return r

    def get_broadside_on_ground(self):
        """
        gets the broadside position on ground at time t=0
        :return: np array 3-d
        """
        k = -self.S_0[2] / self.z_s[2]
        B_gx = float(self.S_0[0] + self.z_s[0] * k)
        B_gy = float(self.S_0[1] + self.z_s[1] * k)
        projection_on_ground = np.array([B_gx, B_gy, 0])
        return projection_on_ground


    # Azimuth and range of a point are the azimuth and range coordinates of closest approach between the target
    # and the satellite. The calculations here performed are valid if the trajectory is a straight line with
    # uniform speed

    def gcs_to_azimuth_range(self, x , y, z):
        # azimuth
        P = np.array([x, y, z]) - self.S_0
        s = np.dot(P, self.velocity / self.abs_v)
        # range
        R = P - (self.velocity / self.abs_v) * s
        range = np.linalg.norm(R)
        return s, range

    def gcs_to_squinted_azimuth_range(self, x, y, z):
        s, r = self.gcs_to_azimuth_range(x, y, z)
        r_c = r / cos(self.forward_squint_angle)
        # point of squinted closest approach
        P0 = np.array([x, y, z]) - self.z_s * r_c
        # projection on velocity unit vector
        s_c = np.dot(P0, self.velocity / self.abs_v)
        return s_c, r_c

    def azimuth_range_to_gcs(self, azimuth, range):
        # point of closest approach # todo test
        v = self.velocity[0:2] / self.abs_v
        v_ortho = np.array([[0, 1], [-1, 0]]) @ v # orthogonal versor
        point = azimuth * v + self.S_0[0:2] # azimuth shift
        point += v_ortho * np.sqrt(self.S_0[2]**2 - range**2) # ground range shift
        # make it 3 d
        point = np.row_stack([point, np.array([0])])
        return point

    def squinted_azimuth_range_to_gcs(self, azimuth_c, range_c):
        return  self.azimuth_range_to_gcs(azimuth_c, range_c)

    def orbital_speed(self):
        """
        ideal circular orbit speed approximation
        :return: the platform estimated speed
        """
        # the platform speed  # gravitational mu # earth radius
        radar_speed = np.sqrt(3.9860044189e14 / (6378e3 + self.S_0[2]))  # m/s
        return radar_speed



