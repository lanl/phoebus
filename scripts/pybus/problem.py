from numpy import *
from geometry import *

gam = 4./3.
do_plot = False

# Choose geometry
a = 0.2
k = pi
geom = Snake(a, k)

# Coordinates
X1min = -1
X1max = 1
X2min = -1
X2max = 1
N1 = 128
N2 = 128
NG = 4
