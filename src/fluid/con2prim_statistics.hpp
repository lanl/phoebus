// © 2022. Triad National Security, LLC. All rights reserved.  This
// program was produced under U.S. Government contract
// 89233218CNA000001 for Los Alamos National Laboratory (LANL), which
// is operated by Triad National Security, LLC for the U.S.
// Department of Energy/National Nuclear Security Administration. All
// rights in the program are reserved by Triad National Security, LLC,
// and the U.S. Department of Energy/National Nuclear Security
// Administration. The Government is granted for itself and others
// acting on its behalf a nonexclusive, paid-up, irrevocable worldwide
// license in this material to reproduce, prepare derivative works,
// distribute copies to the public, perform publicly and display
// publicly, and to permit others to do so.

#ifndef FLUID_CON2PRIM_STATISTICS_HPP_
#define FLUID_CON2PRIM_STATISTICS_HPP_

#include <cstdio>
#include <map>

namespace con2prim_statistics {

class Stats {
 public:
  static std::map<int, int> hist;
  static int max_val;
  static void add(int val) {
    if (hist.count(val) == 0) {
      hist[val] = 1;
    } else {
      hist[val]++;
    }
    max_val = (val > max_val ? val : max_val);
  }
  static void report() {
    auto f = fopen("c2p_stats.txt", "w");
    for (int i = 0; i <= max_val; i++) {
      int cnt = 0;
      if (hist.count(i) > 0) cnt = hist[i];
      fprintf(f, "%d %d\n", i, cnt);
    }
  }

 private:
  Stats() = default;
};

} // namespace con2prim_statistics

#endif
