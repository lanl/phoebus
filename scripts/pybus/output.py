import pickle
from numpy import save

def write_dump(state, n):
  with open('dump_%06i.p' % n, 'wb') as handle:
  #  pickle.dump(state, handle, pickle.HIGHEST_PROTOCOL)
    save(handle, state.prim)
  print("DUMP " + ('dump_%06i.p' % n))
