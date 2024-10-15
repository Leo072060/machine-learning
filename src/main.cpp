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

#define TEST_Mat_sort_row

int main()
{

    string dataFileName = "../test/random_matrix.csv";
    csv_Loader<double> loader;
    Mat<double> DATA = loader.load_matrix(dataFileName);
    display_rainbow(DATA);
    cout<<"FLAG"<<endl;

    // test LinearRegression
    #ifdef TEST_LinearRegression
    auto train_data = DATA.extract_rows(0,DATA.size_row()*0.7);
    auto test_data  = DATA.extract_rows(DATA.size_row()*0.7,DATA.size_row());
    auto train_x = train_data.extract_columns(0,5);
    auto train_y = train_data.extract_columns(5,6);
    LinearRegression<double> model;
    model.train(train_x,train_y);
    display_rainbow(model.THETAS.read());
    auto test_x = test_data.extract_columns(0,5);
    auto test_y = test_data.extract_columns(5,6);
    auto pred_y = model.predict(test_x);
    RegressionEvaluation RE;
    RE.fit(pred_y,test_y);
    RE.report();
    #endif

    // test void Mat<T>::sort_row(const size_t i, const ORDER order)
    #ifdef TEST_Mat_sort_row
    auto data = DATA;
    display_rainbow(data);
    data.sort_column(0,ASCE);
    display_rainbow(data);
    #endif

    return 0;
}
