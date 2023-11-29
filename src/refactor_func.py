import os

from time_numba import TCS_RATES, TCS_RATES_JACOBIAN, EQUILIBRIUM_RATES, EQUILIBRIUM_RATES_JACOBIAN

def refactor_func_string(func_string, func_name, arr_dims, sparse=False):
    array_string = func_string[36:-19]
    arr_vals = array_string.split(',')
    f_template = f"def {func_name}(t, y, kf, kr):\n\tarr = np.zeros({arr_dims})\n"
    row_idx = -1
    col_idx = 0
    for arr_val in arr_vals:
        col_idx += 1
        clean_val = arr_val.strip()
        if clean_val.startswith('['):
            row_idx += 1
            col_idx = 0
            clean_val = clean_val[1:]
        if clean_val.endswith(']]'):
            clean_val = clean_val[:-1]
        if clean_val.endswith(']]'):
            clean_val = clean_val[:-1]
        if clean_val == '0]':
            clean_val = clean_val[:-1]
        if clean_val == '0' and sparse:
            continue
        f_template += f'\tarr[{row_idx}, {col_idx}] = {clean_val}\n'
    f_template += '\treturn arr.reshape(-1)\n\n'
    return f_template

if __name__ == "__main__":
    # Refactor lambdas into functions
    func_strings = [TCS_RATES, TCS_RATES_JACOBIAN, EQUILIBRIUM_RATES, EQUILIBRIUM_RATES_JACOBIAN]
    func_names = ['tcs_rates', 'tcs_rates_jac', 'eq_rates', 'eq_rates_jac']
    arr_shapes = [(29, 1), (29, 41), (37, 1), (37, 104)]
    module_template = "import numpy as np\n\n"
    for i in range(4):
        module_template += refactor_func_string(func_strings[i], func_names[i], arr_shapes[i])
    with open(os.path.join(os.getcwd(), 'src/refactored.py'), 'w') as f:
        f.writelines(module_template)
    # Remove lines that assign 0
    module_template = "import numpy as np\n\n"
    for i in range(4):
        module_template += refactor_func_string(func_strings[i], func_names[i]+'_sp', arr_shapes[i], sparse=True)
    with open(os.path.join(os.getcwd(), 'src/refactored_sparse.py'), 'w') as f:
        f.writelines(module_template)
