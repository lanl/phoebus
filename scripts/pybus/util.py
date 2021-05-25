import sys

def FAIL(message):
  print(message)
  sys.exit()

def delta(mu, nu):
  if mu == nu:
    return 1
  else:
    return 0
