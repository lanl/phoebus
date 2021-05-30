import sys

def FAIL(message):
  print(message)
  sys.exit()

def REQUIRE(condition, message):
  if not condition:
    FAIL(message)

def delta(mu, nu):
  if mu == nu:
    return 1
  else:
    return 0
