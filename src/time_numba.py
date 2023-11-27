import timeit

from numba import njit
import numpy as np

# From TwoComponentSystem.
# The matrix has free_symbols: y[0] .. y[40].
SYMBOLIC_RATES = """lambda t, y: np.array([ \
    [500.0*y[3]*y[4]], [100000000.0*y[0]*y[6]], [0.01*y[5]*y[8]], [170000.0*y[10]*y[4]], \
    [100000000.0*y[12]*y[9]], [0.01*y[11]*y[8]], [0.0001*y[14]*y[4]], [100000000.0*y[12]*y[13]], \
    [170000.0*y[16]*y[4]], [100000000.0*y[15]*y[18]], [0.01*y[17]*y[8]], [0.0001*y[20]*y[4]], \
    [100000000.0*y[18]*y[19]], [170000.0*y[22]*y[4]], [100000000.0*y[21]*y[24]], \
    [0.01*y[23]*y[8]], [0.0001*y[26]*y[4]], [100000000.0*y[24]*y[25]], [170000.0*y[28]*y[4]], \
    [100000000.0*y[27]*y[30]], [0.01*y[29]*y[8]], [0.0001*y[32]*y[4]], \
    [100000000.0*y[30]*y[31]], [500.0*y[34]*y[4]], [100000000.0*y[33]*y[36]], [0.01*y[35]*y[8]], \
    [500.0*y[38]*y[4]], [100000000.0*y[37]*y[40]], [0.01*y[39]*y[8]]]).reshape(-1) \
    """

# The matrix has free_symbols: y[0] .. y[40].
SYMBOLIC_RATES_JACOBIAN = """lambda t, y: np.array([ \
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
     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01*y[8], 0]]) \
    """


def time_jitter(function_name, function_code):
    """Time how long it takes to run the function code, to jit it, and to run
    the jitted function code.
    """
    t = 2.0
    y = np.arange(41.0)
    f = eval(function_code, {'np': np}, {})
    f_jit = njit(f, error_model='numpy')

    time = timeit.timeit(lambda: f(t, y), number=1000)

    time_to_jit = timeit.timeit(lambda: f_jit(t, y), number=1)  # 1st time will JIT f_jit
    time_jitted = timeit.timeit(lambda: f_jit(t, y), number=1000)
    print(f"{function_name}: 1000x {time=}, 1x {time_to_jit=}, 1000x {time_jitted=}")


def time_symbolic_rates():
    time_jitter('RATES', SYMBOLIC_RATES)
    time_jitter('RATES_JACOBIAN', SYMBOLIC_RATES_JACOBIAN)


if __name__ == '__main__':
    time_symbolic_rates()