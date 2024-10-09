#include <iostream>
#include <vector>
#include <filesystem>
#include <exception>
#include <bitset>

#include "dataLoader.h"
#include "numatrix.h"

using namespace std;

int main()
{
  
      string csvFileName = "../test/testData.csv";
      csv_dataLoader<double> csvLoader;
      csvLoader.with_which_name =  WITH_NAME;
      Mat<double> M = csvLoader.load_matrix(csvFileName);
      display_rainbow(M,WITH_COLNAME & WITH_ROWNAME);

      auto M_SiO2 = M.extract_columns(0, 1);
      display_rainbow(M_SiO2,WITH_COLNAME&WITH_ROWNAME);

      auto M_rowNames = M.extract_rowNames();
      display_rainbow(M_rowNames);

      auto M_colNames = M.extract_colNames();
      display_rainbow(M_colNames);

      auto M2 = M;
      M2.transpose();
      display_rainbow(M2, WITH_NAME);

      string csvFileName1 = "../test/random_matrix.csv";
      csv_dataLoader<double> csvLoader1;
      csvLoader1.with_which_name = WITHOUT_NAME;
      Mat<double> Random_M = csvLoader1.load_matrix(csvFileName1);
      display_rainbow(Random_M);

      cout << det(Random_M) << endl;
      auto ret = LU(Random_M);
      auto L = ret["L"];    display_rainbow(L);
      auto U = ret["U"];    display_rainbow(U);

      Mat<double> tmp(10, 1);
      tmp.concat_horizontal(U);

      tmp = Mat<double>(1, 10);
      tmp.concat_vertical(U);

      display_rainbow(tmp);
    tmp *=10;
    display_rainbow(tmp+55);

      system("pause"); 
    return 0;
}
