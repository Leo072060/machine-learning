#include<iostream>
#include<stdexcept>
#include<cmath>

#include"mat/mat.h"
#include"kits/managed.h"

using namespace std;

template<typename T = double>
class RegressionEvaluation:public ManagedClass
{
public:
    RegressionEvaluation():administrator(),
                           TARGET_Y(administrator),PRED_Y(administrator),
                           target_y_MINUS_pred_y(administrator),target_y_MINUS_mean_target_y(administrator),
                           MAE(administrator),
                           MSE(administrator),
                           RMSE(administrator),
                           MAPE(administrator),
                           R2(administrator) {}
public:
    void fit(const Mat<T>& pred_y, const Mat<T>& target_y);
    void report();
    T mean_absolute_error();
    T mean_squared_error();
    T root_mean_squared_error();
    T mean_absolute_percentage_error();
    T r2_score();

private:
    mutable bool isFitted = false;
    Administrator administrator;
    mutable ManagedVal<Mat<T>> TARGET_Y;
    mutable ManagedVal<Mat<T>> PRED_Y;
    mutable ManagedVal<Mat<T>> target_y_MINUS_pred_y;
    mutable ManagedVal<Mat<T>> target_y_MINUS_mean_target_y;
    mutable ManagedVal<T> MAE;
    mutable ManagedVal<T> MSE;
    mutable ManagedVal<T> RMSE;
    mutable ManagedVal<T> MAPE;
    mutable ManagedVal<T> R2;
};

#pragma region function defination

#pragma region member functions
template<typename T>
void RegressionEvaluation<T>::fit(const Mat<T>& pred_y, const Mat<T>& target_y)
{
    if (pred_y.size_column() != 1)
		throw invalid_argument("Error: Matrix pred_y must be single-column matrix.");
    if (target_y.size_column() != 1)
		throw invalid_argument("Error: Matrix target_y must be single-column matrix.");
    this->refresh();
    this->record(TARGET_Y,target_y);
    this->record(PRED_Y,pred_y);
    // calculate
    Mat<T> tmp(target_y.size_row(),1);
    // calculate target_y_MINUS_pred_y
    for(size_t i = 0; i < target_y.size_row(); ++i)
        tmp.iloc(i,0) = target_y.iloc(i,0) - target_y.iloc(i,0);
    this->record(target_y_MINUS_pred_y,tmp);
    // calculate target_y_MINUS_mean_target_y
    for(size_t i = 0; i < target_y.size_row(); ++i)
        tmp.iloc(i,0) = target_y.iloc(i,0) - mean(TARGET_Y.read());
    this->record(target_y_MINUS_mean_target_y,tmp);
    isFitted = true;
}   
template<typename T>
void RegressionEvaluation<T>::report()
{
    if(!isFitted) throw runtime_error("Error: The model must be fitted before generating the report.");
    cout<<"\tLinear Regression Model Performance Report\n";
    cout<<"mean absolute error"           <<"\t"<<mean_absolute_error()           <<endl;
    cout<<"mean squared error"            <<"\t"<<mean_squared_error()            <<endl;
    cout<<"root mean squared error"       <<"\t"<<root_mean_squared_error()       <<endl;
    cout<<"mean absolute percentage error"<<"\t"<<mean_absolute_percentage_error()<<endl;
    cout<<"r2 score"                      <<"\t"<<r2_score()                      <<endl;
}
template<typename T>
T RegressionEvaluation<T>::mean_absolute_error()
{
    if(MAE.readable()) return MAE.read();
    if(!isFitted) throw runtime_error("Error: The model must be fitted before generating mean absolute error.");
    this->record(MAE,mean(abs(target_y_MINUS_pred_y.read())));
    return MAE.read();
}
template<typename T>
T RegressionEvaluation<T>::mean_squared_error()
{
    if(MSE.readable()) return MSE.read();
    if(!isFitted) throw runtime_error("Error: The model must be fitted before generating mean squared error.");
    this->record(MSE,mean(power(target_y_MINUS_pred_y.read(),2)));
    return MSE.read();
}
template<typename T>
T RegressionEvaluation<T>::root_mean_squared_error()
{
    if(RMSE.readable()) return RMSE.read();
    if(!isFitted) throw runtime_error("Error: The model must be fitted before generating root mean squared error.");
    this->record(RMSE,pow(mean_squared_error(),0.5));
    return RMSE.read();
}
template<typename T>
T RegressionEvaluation<T>::mean_absolute_percentage_error()
{
    if(MAPE.readable()) return MAPE.read();
    if(!isFitted) throw runtime_error("Error: The model must be fitted before generating mean absolute percentage error.");
    this->record(MAPE,mean(abs(target_y_MINUS_pred_y.read()/TARGET_Y)));
    return MAPE.read();
}
template<typename T>
T RegressionEvaluation<T>::r2_score()
{
    if(R2.readable()) return R2.read();
    if(!isFitted) throw runtime_error("Error: The model must be fitted before generating r2 score.");
    this->record(R2,1-sum(power(target_y_MINUS_pred_y.read(),2))/sum(power(target_y_MINUS_mean_target_y.read(),2)));
    return R2.read();
}
#pragma endregion

#pragma endregion