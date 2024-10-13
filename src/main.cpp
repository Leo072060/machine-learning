#include <iostream>
#include <vector>
#include <filesystem>
#include <exception>
#include <bitset>

#include"kits/loader.h"
#include"mat/mat.h"
#include"ML/linearModel.h"
#include"ML/evaluation.h"

using namespace std;

#define DEBUG

int main()
{

    string csvFileName = "../test/regression_dataset.csv";
    csv_Loader<double> loader;
    loader.with_which_name = WITH_COLNAME;
    Mat<double> data = loader.load_matrix(csvFileName);
    auto train_data = data.extract_rows(0,data.size_row()*0.7);
    auto test_data  = data.extract_rows(data.size_row()*0.7,data.size_row());

    auto train_x = train_data.extract_columns(0,5);
    auto train_y = train_data.extract_columns(5,6);
    LinearRegression<double> model;
    model.train(train_x,train_y);
    display_rainbow(model.THETAS.read());

    auto test_x = test_data.extract_columns(0,5);
    auto test_y = test_data.extract_columns(5,6);
    // display_rainbow(test_x);
    auto pred_y = model.predict(test_x);
    RegressionEvaluation RE;
    RE.fit(pred_y,test_y);
    RE.report();
    return 0;
}
