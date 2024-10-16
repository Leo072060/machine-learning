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

#define TEST_LogisticRegression

int main()
{

    cout<<"FLAG"<<endl;
    string dataFileName = "../test/random_matrix.csv";
    csv_Loader<double> loader;
    Mat<double> data = loader.load_matrix(dataFileName);
    display_rainbow(data);


    string regression_dataFileName = "../test/regression_data.csv";
    csv_Loader<double> loader_regression_data;
    loader_regression_data.with_which_name = WITH_COLNAME;
    Mat<double> regression_data = loader_regression_data.load_matrix(regression_dataFileName);

    string binary_classiffication_dataFileName = "../test/binary_classification_data.csv";
    csv_Loader<double> loader_binary_classiffication_data;
    loader_binary_classiffication_data.with_which_name = WITH_NAME;
    Mat<double> binary_classiffication_data = loader_binary_classiffication_data.load_matrix(binary_classiffication_dataFileName);
    Mat<string> labels_binary_classiffication_data = binary_classiffication_data.extract_rowNames();
    display_rainbow(binary_classiffication_data);
    display_rainbow(labels_binary_classiffication_data);


    // test void Mat<T>::sort_row(const size_t i, const ORDER order)
    #ifdef TEST_Mat_sort_row
    auto _data = data;
    display_rainbow(_data);
    _data.sort_column(0,ASCE);
    display_rainbow(_data);
    _data.sort_row(3,ASCE);
    display_rainbow(_data);
    #endif

    // test LinearRegression
    #ifdef TEST_LinearRegression
    auto train_data = data.extract_rows(0,data.size_row()*0.7);
    auto test_data  = data.extract_rows(data.size_row()*0.7,data.size_row());
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

    // test LogisticRegression
    #ifdef TEST_LogisticRegression
    auto train_x = binary_classiffication_data.extract_rows(0, binary_classiffication_data.size_row()*0.7);
    auto train_y = labels_binary_classiffication_data.extract_rows(0, binary_classiffication_data.size_row()*0.7);
    auto test_x = binary_classiffication_data.extract_rows(binary_classiffication_data.size_row()*0.7,binary_classiffication_data.size_row());
    auto test_y = labels_binary_classiffication_data.extract_rows(binary_classiffication_data.size_row()*0.7,binary_classiffication_data.size_row());
    LogisticRegression<double> model;
    model.train(train_x,train_y);
    Mat<string> pred_y = model.predict(test_x);
    display_rainbow(concat_horizontal(pred_y, test_y));
    ClassificationEvaluation<double> classificationEvalution;
    classificationEvalution.fit(pred_y, test_y);
    classificationEvalution.report();

    #endif

    return 0;
}


