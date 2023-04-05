# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from extra.radarModel import Radar
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer


def testRadarCoordinates(a: int, b: int, p0: np.array):
    """ :param a alpha angle rotation on xy plane
        :param b beta angle rotation about trajectory respect to a down looking radar
        :param p0 numpy array in the form (x,y,z) initial position of the radar
    """
    # radar class under test
    radar = Radar()
    # radar position
    radar.setInitialPosition(p0)
    # down looking radar rotated on 2 axes
    radar.setRotation(b * np.pi / 180, a * np.pi / 180)
    # creating only one point in time
    radar.time(0, 0, 1)
    # print basis matrices
    print("from local to global\n", radar.Bs2c)
    print("from global to local\n", radar.Bc2s)
    # print radar position
    print("radar position\n", radar.pos)

    # draw basis vectors
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d", proj_type='ortho'))
    fig.set_figheight(8)
    fig.set_figwidth(8)
    # standard basis
    ax.quiver3D(0, 0, 0, 1, 0, 0, colors='r')
    ax.quiver3D(0, 0, 0, 0, 1, 0, colors='g')
    ax.quiver3D(0, 0, 0, 0, 0, 1, colors='b')

    # radar basis
    ax.quiver3D(*(radar.pos[0].tolist()), *(radar.Bc2s[0, :].tolist()), colors='r')
    ax.quiver3D(*(radar.pos[0].tolist()), *(radar.Bc2s[1, :].tolist()), colors='g')
    ax.quiver3D(*(radar.pos[0].tolist()), *(radar.Bc2s[2, :].tolist()), colors='b')

    # radar position
    ax.quiver3D(0, 0, 0, *radar.pos[0].tolist(), colors='grey', arrow_length_ratio=.1)

    # radar broadside on ground
    p = radar.getBroadsideOnGround()[0]
    ax.plot(*p.tolist(), markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=1)

    # plot settings
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_zlim(0, 6)
    ax.grid(True)
    # ax.set_axis_off()
    try:
        ax.set_aspect(aspect='equal')
    except NotImplementedError:
        print("nope, aspect ratio setter still not implemented")
        pass

    # plt.show()


def testAntennaProjection(a: int, b: int, p0: np.array):
    """ tests the default antenna projection on ground for a satellite positioned accordingly to the parameters (to
    be compared to a closed form solution)
    :param a alpha angle rotation on xy plane
    :param b beta angle rotation about trajectory respect to a down looking radar
    :param p0 numpy array in the form (x,y,z) initial position of the radar
    """
    # radar class under test
    radar = Radar()
    # radar position
    radar.setInitialPosition(p0)
    # down looking radar rotated on 2 axes
    radar.setRotation(b * np.pi / 180, a * np.pi / 180)
    # creating only one point in time
    radar.time(0, 0, 1)
    # print basis matrices
    print("from local to global\n", radar.Bs2c)
    print("from global to local\n", radar.Bc2s)
    # print radar position
    print("radar position\n", radar.pos)

    # antenna projection on ground

    # 1 projection limits
    x = np.linspace(-10, 10, 100)  # min max steps
    y = np.linspace(-10, 10, 100)
    # 2 meshgrid
    X, Y = np.meshgrid(x, y)
    # 3 malloc for gain matrix
    G = np.zeros_like(Y)
    # 4 fill the matrix
    for i in np.arange(X.shape[0]):
        for j in np.arange(X.shape[1]):
            P = np.array((X[i, j], Y[i, j], 0))
            G[i, j] = radar.rangeGain(P)[1]
    # 5 plot the ground image obtained
    fig, ax = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(9)

    plt.contourf(Y, X, G, 1, cmap='jet')
    plt.colorbar()

    ax.set_xlabel('y ground [m]')
    ax.set_ylabel('x ground [m]')
    # ax.set_xlim(-.1,.1)
    # ax.set_ylim(-.1,.1)
    ax.set_aspect(1)
    ax.grid()
    # plt.show()


def testCoordinatesAndPattern(a: int, b: int, p0: np.array):
    """ :param a alpha angle rotation on xy plane
        :param b beta angle rotation about trajectory respect to a down looking radar
        :param p0 numpy array in the form (x,y,z) initial position of the radar
    """
    # radar class under test
    radar = Radar()
    # radar position
    radar.setInitialPosition(p0)
    # down looking radar rotated on 2 axes
    radar.setRotation(b * np.pi / 180, a * np.pi / 180)
    # creating only one point in time
    radar.time(0, 0, 1)
    # print basis matrices
    print("from local to global\n", radar.Bs2c)
    print("from global to local\n", radar.Bc2s)
    # print radar position
    print("radar position\n", radar.pos)

    # draw basis vectors
    fig, ax = plt.subplots(subplot_kw=dict(projection="3d", proj_type='ortho'))
    fig.set_figheight(8)
    fig.set_figwidth(8)
    # standard basis
    ax.quiver3D(0, 0, 0, 1, 0, 0, colors='r')
    ax.quiver3D(0, 0, 0, 0, 1, 0, colors='g')
    ax.quiver3D(0, 0, 0, 0, 0, 1, colors='b')

    # radar basis
    ax.quiver3D(*(radar.pos[0].tolist()), *(radar.Bc2s[0, :].tolist()), colors='r')
    ax.quiver3D(*(radar.pos[0].tolist()), *(radar.Bc2s[1, :].tolist()), colors='g')
    ax.quiver3D(*(radar.pos[0].tolist()), *(radar.Bc2s[2, :].tolist()), colors='b')

    # radar position
    ax.quiver3D(0, 0, 0, *radar.pos[0].tolist(), colors='grey', arrow_length_ratio=.1)

    # radar broadside on ground
    p = radar.getBroadsideOnGround()[0]
    ax.plot(*p.tolist(), markerfacecolor='k', markeredgecolor='k', marker='o', markersize=5, alpha=1)

    # plot settings
    ax.set_xlim(-2, 6)
    ax.set_ylim(-2, 6)
    ax.set_zlim(0, 6)
    ax.grid(True)
    # ax.set_axis_off()
    try:
        ax.set_aspect(aspect='equal')
    except NotImplementedError:
        print("nope, aspect ratio setter still not implemented")
        pass

    # antenna projection on ground

    # 1 projection limits
    x = np.linspace(-2, 6, 100)  # min max steps
    y = np.linspace(-2, 6, 100)
    # 2 meshgrid
    X, Y = np.meshgrid(x, y)
    # 3 malloc for gain matrix
    G = np.zeros_like(Y)
    # 4 fill the matrix
    for i in np.arange(X.shape[0]):
        for j in np.arange(X.shape[1]):
            P = np.array((X[i, j], Y[i, j], 0))
            G[i, j] = radar.rangeGain(P)[1]

    ax.contour(x, y, G, 1, color='magenta', alpha=0.9)


def testPatternAutoCentering(a: int, b: int, p0: np.array):
    """ :param a alpha angle rotation on xy plane
        :param b beta angle rotation about trajectory respect to a down looking radar
        :param p0 numpy array in the form (x,y,z) initial position of the radar
    """
    # radar class under test
    radar = Radar()
    # radar position
    radar.setInitialPosition(p0)
    # down looking radar rotated on 2 axes
    radar.setRotation(b * np.pi / 180, a * np.pi / 180)

    # creating only one point in time
    radar.time(0, 0, 1)
    # print basis matrices
    print("from local to global\n", radar.Bs2c)
    print("from global to local\n", radar.Bc2s)
    # print radar position
    print("radar position\n", radar.pos)

    # broadside center on ground
    center = radar.getBroadsideOnGround()[0]

    # 1 projection limits
    x = np.linspace(center[0] - 10, center[0] + 10, 100)  # min max steps
    y = np.linspace(center[1] - 10, center[1] + 10, 100)
    # 2 meshgrid
    X, Y = np.meshgrid(x, y)
    # 3 malloc for gain matrix
    G = np.zeros_like(Y)
    # 4 fill the matrix
    for i in np.arange(X.shape[0]):
        for j in np.arange(X.shape[1]):
            P = np.array((X[i, j], Y[i, j], 0))
            G[i, j] = radar.rangeGain(P)[1]

    # 5 plot the ground image obtained
    fig, ax = plt.subplots()
    fig.set_figheight(9)
    fig.set_figwidth(9)

    plt.contourf(Y, X, G, 1, cmap='jet')
    plt.colorbar()

    # 6 theoretical points
    points = pointsOnGroundPattern(a * np.pi / 180, b * np.pi / 180, p0, radar)
    print("the points are: ", points)
    ax.plot(points[:, 1], points[:, 0], color='k')

    ax.set_xlabel('y ground [m]')
    ax.set_ylabel('x ground [m]')
    # ax.set_xlim(-.1,.1)
    # ax.set_ylim(-.1,.1)
    ax.set_aspect(1)
    ax.grid()


def pointsOnGroundPattern(a: int, b: int, p0: np.ndarray, radar: Radar) -> np.ndarray:
    """ :param a alpha angle rotation on xy plane
        :param b beta angle rotation about trajectory respect to a down looking radar
        :param p0 numpy array in the form (x,y,z) initial position of the radar
        :param radar, it's a Radar
    """
    tx = radar.antenna.lamda / (2 * radar.antenna.L)
    ty = radar.antenna.lamda / (2 * radar.antenna.W)
    pointlist = np.zeros((5, 2))
    i = 0
    for (u, v) in np.array([(1, 1), (-1, 1), (-1, -1), (1, -1)]):
        r = -p0[2] / (v * np.sin(b) * np.sin(ty) - np.cos(b) * np.sqrt(1 - np.sin(tx) ** 2 - np.sin(ty) ** 2))
        x1 = u * r * np.sin(tx)
        y1 = v * r * np.sin(ty)
        z1 = r * np.sqrt(1 - np.sin(tx) ** 2 - np.sin(ty) ** 2)
        pointlist[i, 0] = np.cos(a) * x1 + np.cos(b) * np.sin(a) * y1 + np.sin(b) * np.sin(a) * z1 + p0[0]
        pointlist[i, 1] = np.sin(a) * x1 - np.cos(b) * np.cos(a) * y1 - np.sin(b) * np.cos(a) * z1 + p0[1]
        i += 1
    pointlist[i] = pointlist[0] # close the trapezoid
    return pointlist


def theoreticalRangeGain(a: int, b: int, p0: np.ndarray, radar: Radar, P: np.ndarray):
    """ :param a alpha angle rotation on xy plane
        :param b beta angle rotation about trajectory respect to a down looking radar
        :param p0 numpy array in the form (x,y,z) initial position of the radar
        :param radar, it's a Radar
        :param P point on ground to test
    """
    # 1 get the pointlist for the trapezoid on ground
    [A, B, C, D] = pointsOnGroundPattern(a*np.pi/180, b*np.pi/180, p0 , radar)[0:4] # the last point is copied on the 5th position also
    # 2  find the unit vectors of the two side limit lines
    d_AD = (D - A) / np.linalg.norm(D - A)
    d_BC = (C - B) / np.linalg.norm(C - B)
    # azimuth versor
    sx = np.array((np.cos(a),np.sin(a)))
    # 3 find the first parameter r
    r = ( (P[1]-A[1])*d_AD[0] - (P[0]-A[0])*d_AD[1] ) / ( sx[0]*d_AD[1] - sx[1]*d_AD[0] )
    # 4 find P_A
    P_A = P + sx*r
    # 5 find the second parameter r
    r = ( (P[1]-B[1])*d_BC[0] - (P[0]-B[0])*d_BC[1] ) / ( sx[0]*d_BC[1] - sx[1]*d_BC[0] )
    # 5 find P_B
    P_B = P + sx*r
    # 6 parameter k
    k = np.dot(P_A - A, d_AD)
    # 7 position of P respect to sx
    P_sx = np.dot(P, sx)
    P_Asx = np.dot(P_A, sx)
    P_Bsx = np.dot(P_B, sx)
    # out init
    gain = 0
    # 8 true condition
    if ((P_sx >= P_Bsx) and (P_sx <= P_Asx)) and ((k >= 0) and (k <= np.linalg.norm(C - B))):
        gain = 1
    range = np.linalg.norm(np.array((P[0],P[1],0))-p0)
    return (range, gain)

def testRangeGain(a: int, b: int, p0: np.array, P):
    """ tests the range gain method of the radar class (to be compared with closed form solution)
        :param a alpha angle rotation on xy plane
        :param b beta angle rotation about trajectory respect to a down looking radar
        :param p0 numpy array in the form (x,y,z) initial position of the radar
        :param P point under test on ground np array in the form [x,y,z]
    """
    # radar class under test
    radar = Radar()
    # radar position
    radar.setInitialPosition(p0)
    # down looking radar rotated on 2 axes
    radar.setRotation(b * np.pi / 180, a * np.pi / 180)

    # creating some points in time
    radar.time(0, .001, 1000) # thousand points .001 second
    # print basis matrices
    print("from local to global\n", radar.Bs2c)
    print("from global to local\n", radar.Bc2s)
    # print radar position
    # print("radar position\n", radar.pos)

    (rangesim, gainsim) = radar.rangeGain(P)
    fig, ax = plt.subplots(1,2)
    fig.set_figheight(5)
    fig.set_figwidth(10)
    ax[0].plot(radar.t, gainsim)
    ax[1].plot(radar.t, rangesim)

    # theoretical range gain
    t = np.linspace(0,0.001,1000)
    P_t = np.zeros((1000,3))
    for i in np.arange(t.shape[0]):
        P_t[i,:] = (radar.velocity * t[i]) * radar.Bc2s[0,:] + p0
    ran_t = np.zeros_like(t)
    gain_t = np.zeros_like(t)
    for i in np.arange(t.shape[0]):
        (ran_t[i],gain_t[i]) = theoreticalRangeGain(a, b, P_t[i,:], radar, P[0:2])

    ax[0].plot(t, gain_t*1000, color='r')
    ax[1].plot(t, ran_t, color ='r')
    ax[0].set_xlabel("time")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("range")
    ax[0].set_ylabel("gain")
    ax[0].legend(["simulation"," theory"])

    #plt.show()



pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    testCoordinatesAndPattern(0, 0, np.array([2, 2, 5]))
    testAntennaProjection(0, 0, np.array((0, 0, 20)))
    testPatternAutoCentering(0, 60, np.array((0, 0, 20)))
    testRangeGain(0, 60, np.array((0, 0, 20)), np.array((3, -35, 0)))

    plt.show()
    # pause
    a = input()
