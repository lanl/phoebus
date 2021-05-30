from numpy import *

# Parameters
tf = 1.
dt_init = 1.e-3
gam = 4./3.
cfl = 0.4
do_plot = False

# Geometry
geometry = "snake"
geom_params = {}
geom_params['a'] = 0#0.2
geom_params['k'] = pi

# Coordinates
X1min = -1
X1max = 1
X2min = -1
X2max = 1
N1 = 32
N2 = 32
NG = 4

# Initial conditions
omega = 0.0 + 3.8780661653218766j
drho = 0.5804294924639213
dug = 0.7739059899518947
dv1 = 0.1791244302079596
dv2 = 0.1791244302079596
rho0 = 1.0
ug0 = 1.0
v10 = 0.0
v20 = 0.0
k1 = 2.*pi
k2 = 2.*pi
amp = 1.e-3
