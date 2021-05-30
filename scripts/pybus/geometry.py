from numpy import *

from coordinates import *

class Geometry:
  def __init__(self):
    print("Initializing geometry... ", end='', flush=True)
    for i in range(N1TOT):
      for j in range(N2TOT):
        for loc in range(Location.SIZE):
          self.gcov[i,j,loc,:,:] = self.get_gcov(loc, i, j)
          self.gcon[i,j,loc,:,:] = self.get_gcon(loc, i, j)
          self.alpha[i,j,loc] = self.get_lapse(loc, i, j)
          self.beta[i,j,loc,:] = self.get_shift(loc, i, j)
          self.dgcov[i,j,loc,:,:,:] = self.get_dgcov(loc, i, j)
    print("done!")

  def get_gcon(self, loc, i, j):
    X = getX(loc, i, j)
    return self.get_gcon_X(X)

  def get_gcon_X(self, X):
    gcov = self.get_gcov_X(X)
    return linalg.inv(gcov)

  def get_dgcov(self, loc, i, j):
    X = getX(loc, i, j)
    return self.dgcov_X(X)

  def dgcov_X(self, X):
    dgcov = zeros([4,4,4])
    eps = 1.e-5
    X0 = zeros(4)
    X1 = zeros(4)
    for lam in range(4):
      for mu in range(4):
        X0[mu] = X[mu]
        X1[mu] = X[mu]
      X0[lam] -= eps
      X1[lam] += eps
      gcov0 = self.get_gcov_X(X0)
      gcov1 = self.get_gcov_X(X1)
      for mu in range(4):
        for nu in range(4):
          dgcov[mu,nu,lam] = (gcov1[mu,nu] - gcov0[mu,nu])/(X1[lam] - X0[lam])
    return dgcov

  def dgcon(self, loc, i, j):
    X = getX(loc, i, j)
    return self.dgcon_X(X)

  def dgcon_X(self, X):
    dgcon = zeros([4,4,4])
    eps = 1.e-5
    X0 = zeros(4)
    X1 = zeros(4)
    for lam in range(4):
      for mu in range(4):
        X0[mu] = X[mu]
        X1[mu] = X[mu]
      X0[lam] -= eps
      X1[lam] += eps
      gcon0 = self.gcon(X0)
      gcon1 = self.gcon(X1)
      for mu in range(4):
        for nu in range(4):
          dgcon[mu,nu,lam] = (gcon1[mu,nu] - gcon0[mu,nu])/(X1[lam] - X0[lam])
    return dgcon

  def Jinv(self, loc, i, j):
    X = getX(loc, i, j)
    return self.Jinv_X(X)

  def Jinv_X(self, X):
    J = self.J_X(X)
    return linalg.inv(J)

  gcov = zeros([N1TOT, N2TOT, Location.SIZE, 4, 4])
  gcon = zeros([N1TOT, N2TOT, Location.SIZE, 4, 4])
  alpha = zeros([N1TOT, N2TOT, Location.SIZE])
  beta = zeros([N1TOT, N2TOT, Location.SIZE, 4])
  dgcov = zeros([N1TOT, N2TOT, Location.SIZE, 4, 4, 4])

class Snake(Geometry):
  def __init__(self, geom_params):
    self.a = geom_params['a']
    self.k = geom_params['k']
    super().__init__()

  def get_gcov(self, loc, i, j):
    X = getX(loc, i, j)
    return self.get_gcov_X(X)

  def get_gcov_X(self, X):
    gcov = zeros([4, 4])
    delta = self.a*self.k*cos(self.k*X[1])
    gcov[0,0] = -1
    gcov[1,1] = 1
    gcov[2,1] = -delta
    gcov[1,2] = -delta
    gcov[2,2] = delta**2 + 1
    gcov[3,3] = 1
    return gcov

  def J(self, loc, i, j):
    X = getX(loc, i, j)
    return self.J_X(X)

  def J_X(self, X):
    J = zeros([4, 4])
    for mu in range(4):
      J[mu,mu] = 1.
    J[2,1] = -self.a*self.k*cos(self.k*X[1])

    return J

  def get_lapse(self, loc, i, j):
    return 1.;

  def get_shift(self, loc, i, j):
    shift = zeros(4)
    return shift

# Select geometry and provide to code
if geometry == "snake":
  geom = Snake(geom_params)
else:
  FAIL("Geometry \"" + geometry + "\" not recognized!")
