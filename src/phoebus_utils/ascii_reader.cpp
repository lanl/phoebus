#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
#include <utils/error_checking.hpp>
#include "ascii_reader.hpp"

using namespace std;
namespace AsciiReader {

void readtable (string filename, std::vector<double> mydata[11]){

  //std::vector<double> mydata[11];
  ifstream myfile(filename.c_str());
 if (!myfile.is_open()){
   stringstream msg;
   msg << "file " << filename << " not found." << std::endl;
   PARTHENON_FAIL(msg);
 }
 int i=0;
 while(!myfile.eof()){
   string line;
   std::getline(myfile, line);
   if(line[0] =='#') continue; //for comments
   stringstream ss(line);
   int col = 0;
   double temp;
   while(ss >> temp) {
     mydata[col].push_back(temp);
     col++;
   }

 }
}
 double calculatemass(std::vector<double> mydata[11]){
 double mass=0;
 for (int i=0; i<=mydata[0].size()-1; i++){
   //mass=mass+4*3.14*(rho[i+1]*pow(rad[i+1],2)+rho[i]*pow(rad[i],2))/2*(rad[i+1]-rad[i]);
   std::vector<double> &rho = mydata[COLUMNS::rho];
   std::vector<double> &rad = mydata[COLUMNS::rad];
   mass=mass+4*3.14*(rho[i+1]*pow(rad[i+1],2)+rho[i]*pow(rad[i],2))/2*(rad[i+1]-rad[i]);
 }
 //std::cout << mass << std::endl;

 return mass;
 }


} //namespace AsciiReader
