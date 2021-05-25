from numpy import *
from enum import Enum

class Location(Enum):
  CENT = 0
  FACE1 = 1
  FACE2 = 2
  CORN = 3

class Geometry:
  #def gcon(self, loc, i, j):
  #  X = getX(loc, i, j)
  #  return self.gcon(X)

  def gcon(self, X):
    gcov = self.gcov(X)
    return linalg.inv(gcov)

  def dgcov(self, loc, i, j):
    X = getX(loc, i, j)
    return self.dgcov(X)

  def dgcov(self, X):
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
      gcov0 = self.gcov(X0)
      gcov1 = self.gcov(X1)
      for mu in range(4):
        for nu in range(4):
          dgcov[mu,nu,lam] = (gcov1[mu,nu] - gcov0[mu,nu])/(X1[lam] - X0[lam])
    return dgcov

  def dgcon(self, loc, i, j):
    X = getX(loc, i, j)
    return self.dgcon(X)

  def dgcon(self, X):
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
    return self.Jinv(X)

  def Jinv(self, X):
    J = self.J(X)
    return linalg.inv(J)

class Snake(Geometry):
  def __init__(self, a, k):
    self.a = a
    self.k = k

  def gcov(self, loc, i, j):
    X = getX(loc, i, j)
    return self.gcov(X)

  def gcov(self, X):
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
    return self.J(X)

  def J(self, X):
    J = zeros([4, 4])
    for mu in range(4):
      J[mu,mu] = 1.
    J[2,1] = -self.a*self.k*cos(self.k*X[1])

    return J
