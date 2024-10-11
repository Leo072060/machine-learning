#include <iostream>
#include <vector>
#include <filesystem>
#include <exception>
#include <bitset>

#include"kits/loader.h"
#include"mat/mat.h"
#include"ML/linearModel.h"

using namespace std;

int main()
{
  
    string csvFileName = "../test/regression_dataset.csv";
    csv_Loader<double> loader;
    loader.with_which_name = WITH_COLNAME;
    Mat<double> M = loader.load_matrix(csvFileName);
    // display_rainbow(M);
    auto x = M.extract_columns(0,5);
    auto y = M.extract_columns(5,6);
    // display_rainbow(x);
    // display_rainbow(y);

    LinearRegression<double> model;
    model.train(x,y);
    auto ret = model.get_trainedParameters();
    cout<<"theta0 : "<<ret["theta0"]<<endl;
    cout<<"theta1 : "<<ret["theta1"]<<endl;
    cout<<"theta2 : "<<ret["theta2"]<<endl;
    cout<<"theta3 : "<<ret["theta3"]<<endl;
    cout<<"theta4 : "<<ret["theta4"]<<endl;
    cout<<"theta5 : "<<ret["theta5"]<<endl;
    // cout<<"theta6 : "<<ret["theta6"]<<endl;
    // cout<<"theta7 : "<<ret["theta7"]<<endl;
    return 0;
}
