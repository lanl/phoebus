from numpy import *

class State:
  def __init__(self, nx1, nx2):
    self.prim_rho = zeros([nx1, nx2])
    self.prim_ug = zeros([nx1, nx2])
    self.prim_v1 = zeros([nx1, nx2])
    self.prim_v2 = zeros([nx1, nx2])

    self.cons_rho = zeros([nx1, nx2])
    self.cons_ug = zeros([nx1, nx2])
    self.cons_v1 = zeros([nx1, nx2])
    self.cons_v2 = zeros([nx1, nx2])

