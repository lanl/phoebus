# © 2022. Triad National Security, LLC. All rights reserved.  This
# program was produced under U.S. Government contract
# 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
# is operated by Triad National Security, LLC for the U.S.  Department
# of Energy/National Nuclear Security Administration. All rights in
# the program are reserved by Triad National Security, LLC, and the
# U.S. Department of Energy/National Nuclear Security
# Administration. The Government is granted for itself and others
# acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
# license in this material to reproduce, prepare derivative works,
# distribute copies to the public, perform publicly and display
# publicly, and to permit others to do so.


def progress_bar(fraction, title="Calculating"):
    import sys

    bar_length = 20
    status = ""
    block = int(round(bar_length * fraction))
    if fraction >= 1.0:
        fraction = 1.0
        status = "\n"
    text = "\r{0}: [{1}]{2}".format(
        title, "#" * block + "-" * (bar_length - block), status
    )
    sys.stdout.write(text)
    sys.stdout.flush()
