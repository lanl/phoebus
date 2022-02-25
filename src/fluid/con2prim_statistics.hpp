#ifndef FLUID_CON2PRIM_STATISTICS_HPP_
#define FLUID_CON2PRIM_STATISTICS_HPP_

#include <cstdio>
#include <map>

namespace con2prim_statistics {

class Stats {
 public:
  static std::map<int,int> hist;
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
    auto f = fopen("c2p_stats.txt","w");
    for (int i = 0; i <= max_val; i++) {
      int cnt = 0;
      if (hist.count(i) > 0) cnt = hist[i];
      fprintf(f, "%d %d\n", i, cnt);
    }
  }
 private:
  Stats() = default;
};

}

#endif