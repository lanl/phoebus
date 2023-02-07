#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <sstream>
#include <fstream>
#include "reader.hpp"


int main(int argc, char *argv[]){

  std::vector<double> mydata[11];
  readtable(argv[1], mydata);
  std::cout << mydata[1][1] << std::endl;
  double mass;
  mass=calculatemass(mydata);
  std::cout << mass << std::endl;

  return 0;

}
