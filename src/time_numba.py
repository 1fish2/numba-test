import platform
import timeit

from numba import njit
import numpy as np

from refactored import tcs_rates, tcs_rates_jac, eq_rates, eq_rates_jac
from refactored_sparse import tcs_rates_sp, tcs_rates_jac_sp, eq_rates_sp, eq_rates_jac_sp

NUMBER = 10_000  # number of times for timeit to run the timed code

# From WCM TwoComponentSystem.
TCS_RATES = """lambda t, y, kf, kr: np.array([ \
    [500.0*y[3]*y[4]], [100000000.0*y[0]*y[6]], [0.01*y[5]*y[8]], [170000.0*y[10]*y[4]], \
    [100000000.0*y[12]*y[9]], [0.01*y[11]*y[8]], [0.0001*y[14]*y[4]], [100000000.0*y[12]*y[13]], \
    [170000.0*y[16]*y[4]], [100000000.0*y[15]*y[18]], [0.01*y[17]*y[8]], [0.0001*y[20]*y[4]], \
    [100000000.0*y[18]*y[19]], [170000.0*y[22]*y[4]], [100000000.0*y[21]*y[24]], \
    [0.01*y[23]*y[8]], [0.0001*y[26]*y[4]], [100000000.0*y[24]*y[25]], [170000.0*y[28]*y[4]], \
    [100000000.0*y[27]*y[30]], [0.01*y[29]*y[8]], [0.0001*y[32]*y[4]], \
    [100000000.0*y[30]*y[31]], [500.0*y[34]*y[4]], [100000000.0*y[33]*y[36]], [0.01*y[35]*y[8]], \
    [500.0*y[38]*y[4]], [100000000.0*y[37]*y[40]], [0.01*y[39]*y[8]]]).reshape(-1) \
    """

# From WCM TwoComponentSystem.
TCS_RATES_JACOBIAN = """lambda t, y, kf, kr: np.array([ \
    [0, 0, 0, 500.0*y[4], 500.0*y[3], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [100000000.0*y[6], 0, 0, 0, 0, 0, 100000000.0*y[0], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0.01*y[8], 0, 0, 0.01*y[5], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 170000.0*y[10], 0, 0, 0, 0, 0, 170000.0*y[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 100000000.0*y[12], 0, 0, 100000000.0*y[9], 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0.01*y[11], 0, 0, 0.01*y[8], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0.0001*y[14], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0001*y[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100000000.0*y[13], 100000000.0*y[12], 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 170000.0*y[16], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170000.0*y[4], 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100000000.0*y[18], 0, 0, 100000000.0*y[15], 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0.01*y[17], 0, 0, 0, 0, 0, 0, 0, 0, 0.01*y[8], 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0.0001*y[20], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0001*y[4], 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100000000.0*y[19], 100000000.0*y[18], 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 170000.0*y[22], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 170000.0*y[4], 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100000000.0*y[24], 0, 0, \
     100000000.0*y[21], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0.01*y[23], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01*y[8], 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0.0001*y[26], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0.0001*y[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100000000.0*y[25], \
     100000000.0*y[24], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 170000.0*y[28], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 170000.0*y[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     100000000.0*y[30], 0, 0, 100000000.0*y[27], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0.01*y[29], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0.01*y[8], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0.0001*y[32], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0.0001*y[4], 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     100000000.0*y[31], 100000000.0*y[30], 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 500.0*y[34], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 500.0*y[4], 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 100000000.0*y[36], 0, 0, 100000000.0*y[33], 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0.01*y[35], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0.01*y[8], 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 500.0*y[38], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 500.0*y[4], 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 100000000.0*y[40], 0, 0, 100000000.0*y[37]], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0.01*y[39], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01*y[8], 0]]).reshape(-1) \
    """

# From WCM Equilibrium.
EQUILIBRIUM_RATES = """lambda t, y, kf, kr: np.array([ \
    [kf[0]*y[1]*y[2] - kr[0]*y[0]], [kf[1]*y[4]*y[5] - kr[1]*y[3]], [kf[2]*y[7]*y[8] - kr[2]*y[6]], \
    [kf[3]*y[10]*y[11] - kr[3]*y[9]], [kf[4]*y[10]*y[13] - kr[4]*y[12]], \
    [kf[5]*y[15]*y[16] - kr[5]*y[14]], [kf[6]*y[18]*y[19] - kr[6]*y[17]], \
    [kf[7]*y[21]*y[22] - kr[7]*y[20]], [kf[8]*y[24]*y[25] - kr[8]*y[23]], \
    [kf[9]*y[27]*y[28] - kr[9]*y[26]], [kf[10]*y[30]*y[31] - kr[10]*y[29]], \
    [kf[11]*y[33]*y[34] - kr[11]*y[32]], [kf[12]*y[36]*y[37] - kr[12]*y[35]], \
    [kf[13]*y[39]*y[40] - kr[13]*y[38]], [kf[14]*y[22]*y[42] - kr[14]*y[41]], \
    [kf[15]*y[10]*y[44] - kr[15]*y[43]], [kf[16]*y[46]*y[47] - kr[16]*y[45]], \
    [kf[17]*y[49]*y[50] - kr[17]*y[48]], [kf[18]*y[52]*y[53] - kr[18]*y[51]], \
    [kf[19]*y[55]*y[56] - kr[19]*y[54]], [kf[20]*y[22]*y[58] - kr[20]*y[57]], \
    [kf[21]*y[60]*y[61] - kr[21]*y[59]], [kf[22]*y[63]*y[64] - kr[22]*y[62]], \
    [kf[23]*y[66]*y[67] - kr[23]*y[65]], [kf[24]*y[66]*y[69] - kr[24]*y[68]], \
    [kf[25]*y[63]*y[71] - kr[25]*y[70]], [kf[26]*y[40]*y[73] - kr[26]*y[72]], \
    [kf[27]*y[75]*y[76]**2.0 - kr[27]**2.0*y[74]], [kf[28]*y[78]*y[79] - kr[28]*y[77]], \
    [kf[29]*y[81]*y[82]**2.0 - kr[29]**2.0*y[80]], [kf[30]*y[84]*y[85] - kr[30]*y[83]], \
    [kf[31]*y[87]*y[88] - kr[31]*y[86]], [kf[32]*y[90]*y[91] - kr[32]*y[89]], \
    [kf[33]*y[93]*y[94] - kr[33]*y[92]], [kf[34]*y[96]*y[97] - kr[34]*y[95]], \
    [kf[35]*y[100]*y[99] - kr[35]*y[98]], [kf[36]*y[102]*y[103] - kr[36]*y[101]]]).reshape(-1) \
    """

# From WCM Equilibrium.
EQUILIBRIUM_RATES_JACOBIAN = """lambda t, y, kf, kr: np.array([ \
    [-kr[0], kf[0] * y[2], kf[0] * y[1], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, -kr[1], kf[1] * y[5], kf[1] * y[4], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, -kr[2], kf[2] * y[8], kf[2] * y[7], 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[3], kf[3] * y[11], kf[3] * y[10], 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kf[4] * y[13], 0, -kr[4], kf[4] * y[10], 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[5], kf[5] * y[16], \
     kf[5] * y[15], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[6], kf[6] * y[19], \
     kf[6] * y[18], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[7], \
     kf[7] * y[22], kf[7] * y[21], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     -kr[8], kf[8] * y[25], kf[8] * y[24], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, -kr[9], kf[9] * y[28], kf[9] * y[27], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, -kr[10], kf[10] * y[31], kf[10] * y[30], 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, -kr[11], kf[11] * y[34], kf[11] * y[33], 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[12], kf[12] * y[37], kf[12] * y[36], 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[13], kf[13] * y[40], \
     kf[13] * y[39], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     kf[14] * y[42], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     -kr[14], kf[14] * y[22], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kf[15] * y[44], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[15], \
     kf[15] * y[10], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[16], \
     kf[16] * y[47], kf[16] * y[46], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     -kr[17], kf[17] * y[50], kf[17] * y[49], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, -kr[18], kf[18] * y[53], kf[18] * y[52], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, -kr[19], kf[19] * y[56], kf[19] * y[55], 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     kf[20] * y[58], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[20], kf[20] * y[22], 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[21], kf[21] * y[61], kf[21] * y[60], 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[22], kf[22] * y[64], \
     kf[22] * y[63], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[23], kf[23] * y[67], \
     kf[23] * y[66], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kf[24] * y[69], 0, -kr[24], \
     kf[24] * y[66], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kf[25] * y[71], 0, 0, 0, 0, 0, 0, \
     -kr[25], kf[25] * y[63], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, kf[26] * y[73], 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, -kr[26], kf[26] * y[40], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     -kr[27] ** 2.0, kf[27] * y[76] ** 2.0, 2.0 * kf[27] * y[75] * y[76] ** 1.0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, -kr[28], kf[28] * y[79], kf[28] * y[78], 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, -kr[29] ** 2.0, kf[29] * y[82] ** 2.0, \
     2.0 * kf[29] * y[81] * y[82] ** 1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, -kr[30], kf[30] * y[85], kf[30] * y[84], 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[31], kf[31] * y[88], kf[31] * y[87], \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[32], kf[32] * y[91], \
     kf[32] * y[90], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[33], kf[33] * y[94], \
     kf[33] * y[93], 0, 0, 0, 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -kr[34], \
     kf[34] * y[97], kf[34] * y[96], 0, 0, 0, 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     -kr[35], kf[35] * y[100], kf[35] * y[99], 0, 0, 0], \
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
     0, -kr[36], kf[36] * y[103], kf[36] * y[102]]]).reshape(-1) \
    """


def time_jit(function_name, function_code):
    """Time how long it takes to run the function code, to JIT compile it, and
    to run the JIT compiled code.
    """
    t = 2.0
    y = np.arange(105.0)
    kf = np.arange(100.0) * 2
    kr = np.arange(100.0) * 10

    if isinstance(function_code, str):
        f = eval(function_code, {'np': np}, {})
    else:
        f = function_code
    f_jit = njit(f, error_model='numpy')

    time = timeit.timeit(lambda: f(t, y, kf, kr), number=NUMBER) / NUMBER

    time_to_jit = timeit.timeit(lambda: f_jit(t, y, kf, kr), number=1)  # 1st time JITs
    time_jitted = timeit.timeit(lambda: f_jit(t, y, kf, kr), number=NUMBER) / NUMBER

    iterations_to_payoff = time_to_jit / (time - time_jitted)  # for JITted function
    print(f"{function_name:27} {time * 1000.0} ms,"
          f" jitting {time_to_jit * 1000.0:,} ms,"
          f" jitted {time_jitted * 1000.0} ms,"
          f" iterations to payoff {iterations_to_payoff:,.0f}")


def time_symbolic_rates():
    time_jit('tcs_rates_sp', tcs_rates_sp)
    time_jit('tcs_rates_jac_sp', tcs_rates_jac_sp)
    time_jit('eq_rates_sp', eq_rates_sp)
    time_jit('eq_rates_jac_sp', eq_rates_jac_sp)

    time_jit('tcs_rates', tcs_rates)
    time_jit('tcs_rates_jac', tcs_rates_jac)
    time_jit('eq_rates', eq_rates)
    time_jit('eq_rates_jac', eq_rates_jac)

    time_jit('2CS RATES', TCS_RATES)
    time_jit('2CS RATES JACOBIAN', TCS_RATES_JACOBIAN)
    time_jit('EQUILIBRIUM RATES', EQUILIBRIUM_RATES)
    time_jit('EQUILIBRIUM RATES JACOBIAN', EQUILIBRIUM_RATES_JACOBIAN)


def environment():
    return f"Python {platform.python_version()} on {platform.platform()}"


if __name__ == '__main__':
    print(f"Numba timings on {environment()}")
    time_symbolic_rates()
