#ifndef READER_HPP_INCLUDED
#define READER_HPP_INCLUDED
#include <string>
#include <vector>

namespace AsciiReader{
namespace COLUMNS {
  constexpr int rad = 0, rho = 1, temp=2, ye = 3, eps = 4, vel = 5, press = 6, rho_adm = 7, P_adm =8, S_adm = 9, Srr_adm = 10;
 }
void readtable(std::string filename, std::vector<double> mydata[11]);
double calculatemass(std::vector<double> mydata[11]);
} //namespace AsciiReader
#endif
