import numpy as np
import pandas as pd
from npreadtext import _loadtxt

# Prepare csv file
a = np.random.rand(100_000, 5)
np.savetxt("test.csv", a, delimiter=",")

print("loadtxt:")
get_ipython().run_line_magic('timeit', 'b = np.loadtxt("test.csv", delimiter=",")')
print("read_csv")
get_ipython().run_line_magic('timeit', 'c = pd.read_csv("test.csv", delimiter=",")')
print("read_csv w/ honest floats")
get_ipython().run_line_magic(
    'timeit', 'c = pd.read_csv("test.csv", delimiter=",", float_precision="round_trip")'
)
print("_loadtxt")
get_ipython().run_line_magic('timeit', 'd = _loadtxt("test.csv", delimiter=",")')
