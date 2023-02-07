#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>
using namespace std;

namespace COLUMNS {
  constexpr int rad = 0, rho = 1, temp=2, ye = 3, eps = 4, vel = 5, press = 6, rho_adm = 7, P_adm =8, S_adm = 9, Srr_adm = 10;
 }

void readtable (string filename, std::vector<double> mydata[11]){

  //std::vector<double> mydata[11];
 ifstream myfile(filename);
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


