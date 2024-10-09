#include <iostream>
#include <vector>
#include <filesystem>
#include <exception>
#include <bitset>

#include"dataLoader.h"
#include"numatrix.h"
#include"linearModel.h"

using namespace std;

int main()
{
  
    string csvFileName = "../test/regression_dataset.csv";
    csv_dataLoader<double> loader;
    loader.with_which_name = WITH_COLNAME;
    Mat<double> M = loader.load_matrix(csvFileName);
    display_rainbow(M);
    auto x = M.extract_columns(0,2);
    auto y = M.extract_columns(2,3);
    display_rainbow(x);
    display_rainbow(y);

    LinearRegression<double> model;
    model.train(x,y);
    auto ret = model.get_trainedParameters();
    cout<<"theta0 : "<<ret["theta0"]<<endl;
    cout<<"theta0 : "<<ret["theta1"]<<endl;
    return 0;
}
