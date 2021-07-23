import numpy as np
import pandas as pd
from npreadtext import _loadtxt

# Prepare csv file
a = np.random.rand(100_000, 5)
np.savetxt("test.csv", a, delimiter=",")

cmds_to_bench = [
    'b = np.loadtxt("test.csv", delimiter=",")',
    'c = pd.read_csv("test.csv", delimiter=",")',
    'd = pd.read_csv("test.csv", delimiter=",", float_precision="round_trip")',
    'e = _loadtxt("test.csv", delimiter=",")',
]

for cmd in cmds_to_bench:
    print(f">>> {cmd}")
    get_ipython().run_line_magic('timeit', cmd)
