"""
This test builds and times simplistic functions with various numbers of lines to
test how Numba JIT time grows with the size of the function body it compiles.

Pass in zero or more function size numbers on the command line.

This makes it easy to measure how the JIT compilation speed grows with the size
of the function its compiling, and easy to test across Python versions.
"""

import sys

from time_numba import environment, time_jit


def build_def(name: str, line_count: int = 10) -> str:
    """Build a simplistic function with the given number of array-element
    assignment lines. It'll be called with float t and ndarrays y, kf, kr.
    """
    lines = [f'def {name}(t, y, kf, kr):',
             '\tarr = np.zeros((10, 10))']

    for i in range(line_count):
        lines.append('\tarr[0, 0] = 0')

    lines.append('\treturn arr')
    return '\n'.join(lines)


def time_variable_length_def(line_count: int = 10):
    name = f'def_{line_count}'
    time_jit(name, build_def(name, line_count), execute=True)


def time_multiple_length_defs(line_counts: list[int]):
    for count in line_counts:
        time_variable_length_def(count)


if __name__ == '__main__':
    line_counts = [int(arg) for arg in sys.argv[1:]]
    if not line_counts:
        line_counts = [10, 100, 500, 1000, 2000, 3000]

    print(f"### Numba timing {line_counts}-line function(s) on {environment()}  ")

    time_multiple_length_defs(line_counts)
