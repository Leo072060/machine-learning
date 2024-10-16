#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#include<stdexcept> 
#include<vector>
#include<random>
#include<set>
#include<cmath>


#include"ML/modelBase.h"
#include"kits/managed.h"
#include"mat/mat.h"
#include"kits/dict.h"
#include"math/numerical_optimization.h"







template<typename T>
class SimpleLinearRegression:public RegressionModelBase<T>
{
public:
	SimpleLinearRegression() :RegressionModelBase<T>(),W(this->administrator), B(this->administrator), M(this->administrator) {}

// * * * * * * * functions * * * * * * * 
public:
	void    train                (const Mat<T>& x, const Mat<T>& y) override; 
	Mat<T>  predict              (const Mat<T>& x) const            override;
	T       predict              (const T& x)	   const;

// * * * * * * * attributes * * * * * * *
public:
	// calculated value
	ManagedVal<T>      W;
	ManagedVal<T>      B;
	ManagedVal<size_t> M;
};

#pragma region function definition

#pragma region member functions

template<typename T>
void SimpleLinearRegression<T>::train(const Mat<T>& x, const Mat<T>& y)
{
	if (x.size_column() != 1 || y.size_column() != 1) 
		throw invalid_argument("Error: Both x and y must be single-column matrices.");
	if (x.size_row() != y.size_row()) 
		throw invalid_argument("Error: Matrix x and y must have the same number of rows.");
	if (x.size_row() < 1) 
		throw invalid_argument("Error: Matrix x must have at least one row.");

	record(M, x.size_row());

	T w			      = 0;
	T w_numerator     = 0;
	T w_denominator   = 0;
	T w_denominator_1 = (x.sum() / M) * x.sum();
	T b				  = 0;
	for (size_t i = 0; i < M; ++i)
	{
		w_numerator	  += (y[i] * (x[i] - x.mean()));
		w_denominator += x[i] * x[i];
	}
	w = (w_numerator / (w_denominator - w_denominator_1));
	for (size_t i = 0; i < M; ++i)
		b += (y[i] - w * x[i]) / M;

	this->record(W, w);
	this->record(B, b);
}
template<typename T>
Mat<T> SimpleLinearRegression<T>::predict(const Mat<T>& x) const
{
	if(x.size_column()!=1)
		throw invalid_argument("Error: Input matrix x must be a single-column matrix for prediction.");
	Mat<T> y(1,x.size_row);
	for (size_t i = 0; i < x.size_row; ++i) 
		y.iloc(i, 0) = x.iloc(i, 0) * W.read() + B.read();
	return y;
}
template<typename T>
T SimpleLinearRegression<T>::predict(const T& x) const
{
	if (!W.readable() || !B.readable())
		throw runtime_error("Error: Weights and biases must be readable before accessing trained parameters.");
	return x * W.read() + B.read();
}
#pragma endregion 

#pragma endregion 







template<typename T>
class LinearRegression :public RegressionModelBase<T>
{
public:
    LinearRegression():RegressionModelBase<T>(),THETAS(this->administrator){};
// * * * * * * * functions * * * * * * *
public:
	void    train  (const Mat<T>& x, const Mat<T>& y) override;
	Mat<T>  predict(const Mat<T>& x) const			override;
// * * * * * * * attributes * * * * * * *
public:
	// model parameters
	double learning_rate = 0.0003;
	size_t batch_size = 100;
	size_t iterations = 1700;
public:
	// calculated value
	ManagedVal<Mat<T>> THETAS;
};

#pragma region function definition

#pragma region member functions

template<typename T>
void LinearRegression<T>::train(const Mat<T>& x, const Mat<T>& y)
{
    Mat<T> ones(x.size_row(),1);
    for(size_t i = 0; i < x.size_row(); ++i) ones.iloc(i,0) = 1;
    Mat<T> w = concat_horizontal(x,ones);
    Mat<T> thetas(1,x.size_column()+1);

    // start training
    for(size_t i = 0; i < iterations; ++i)
    {
#ifdef DEBUG
    cout<<"Training iteration : "<<i+1<<endl;
#endif
    // generate random numbers
    if(x.size_row() < batch_size)
         throw out_of_range("Error: Batch size is larger than the available rows.");
    set<size_t> randomNums;
    random_device rd;
    mt19937 gen(rd()); 
    uniform_int_distribution<> dis(0, x.size_row()-1);
    while (randomNums.size() < batch_size) 
        randomNums.insert(dis(gen)); 
    
    Mat<T> tmp_thetas(thetas);

    for(size_t i = 0; i < w.size_column(); ++i)
    {
        T tmp_theta_i = 0;
        for(auto& e:randomNums)
        {
            tmp_theta_i += learning_rate*((y.iloc_row(e) - dot(thetas,transpose(w.iloc_row(e)))) * w.iloc(e,i)).iloc(0,0);
        }
        tmp_thetas.iloc(0,i) += (tmp_theta_i/batch_size);
    }
    thetas = tmp_thetas;
#ifdef DEBUG
    display_rainbow(thetas);
#endif
    }
    this->record(THETAS,thetas);
}
template<typename T>
Mat<T> LinearRegression<T>::predict(const Mat<T>& x) const
{
    if (x.size_column() + 1 != THETAS.read().size_column()) 
        throw invalid_argument("Error: The input matrix has incompatible dimensions with the model parameters.");
    Mat<T> ones(x.size_row(),1);
    ones.fill(1);
    Mat<T> w = concat_horizontal(x,ones);
	return dot(w,(transpose(THETAS.read())));
}
#pragma endregion 

#pragma endregion 







template<typename T>
class Log_LinearRegression:public LinearRegression<T>
{
public:
    Log_LinearRegression():RegressionModelBase<T>(),THETAS(this->administrator){};
// * * * * * * * functions * * * * * * *
public:
	void    train(const Mat<T>& x, const Mat<T>& y) override;
	Mat<T>  predict(const Mat<T>& x) const			override;

// * * * * * * * attributes * * * * * * *
public:
	// model parameters
	double learning_rate = 0.0003;
	size_t batch_size = 100;
	size_t iterations = 3000;
private:
	// calculated value
	ManagedVal<Mat<T>> THETAS;
};

#pragma region function definition

#pragma region member functions
template<typename T>
void Log_LinearRegression<T>::train(const Mat<T>& x, const Mat<T>& y)
{
    Mat<T> ones(x.size_row(),1);
    ones.fill(1);
    Mat<T> w = concat_horizontal(x,ones);
    Mat<T> thetas(1,x.size_column()+1);

    Mat<T> log_y(y);
    for(size_t i = 0; i < log_y.size_row(); ++i)
    {
        log_y.iloc(i,0) = log(y.iloc(i,0));
    }

    // start training
    for(size_t i = 0; i < iterations; ++i)
    {
    cout<<"Output of the training iteration : "<<i+1<<endl;
    // generate random numbers
    if(x.size_row() < batch_size)
         throw out_of_range("Error: Batch size is larger than the available rows.");
    set<size_t> randomNums;
    random_device rd;
    mt19937 gen(rd()); 
    uniform_int_distribution<> dis(0, x.size_row()-1);
    while (randomNums.size() < batch_size) 
        randomNums.insert(dis(gen)); 
    
    Mat<T> tmp_thetas(thetas);

    for(size_t i = 0; i < w.size_column(); ++i)
    {
        T tmp_theta_i = 0;
        for(auto& e:randomNums)
        {
            tmp_theta_i += learning_rate*((log_y.iloc_row(e) - dot(thetas,transpose(w.iloc_row(e)))) * w.iloc(e,i)).iloc(0,0);
        }
        tmp_thetas.iloc(0,i) += (tmp_theta_i/batch_size);
    }
    thetas = tmp_thetas;
    display_rainbow(thetas);
    }
    this->record(THETAS,thetas);
}
template<typename T>
Mat<T> Log_LinearRegression<T>::predict(const Mat<T>& x) const
{
	if (x.size_column() != 1)
		throw invalid_argument("Error: Input matrix x must be a single-column matrix for prediction.");
    Mat<T> ones(x.size_row(),1);
    ones.fill(1);
    Mat<T> w = concat_horizontal(x,ones);
	return w.dot(transpose(x));
}
#pragma endregion 

#pragma endregion 







template<typename T>
class LogisticRegression : ClassificationModelBase<T>
{
public:
    LogisticRegression() : ClassificationModelBase<T>(),THETAS(this->administrator),LABELS(this->administrator) {}
public:
    void		  train                (const Mat<T>& x, const Mat<string>& y) override;
	Mat<string>   predict              (const Mat<T>& x) const                 override;
    Mat<T>        predict_probabilities(const Mat<T>& x) const; 
    static Mat<T> predict_probabilities(const Mat<T>& x,const Mat<T>& thetas);

public:
	// model parameters
	double learning_rate = 0.0003;
	size_t batch_size = 70;
	size_t iterations = 3000;
private:
	// calculated value
	mutable ManagedVal<Mat<T>> THETAS;
    mutable ManagedVal<Mat<string>> LABELS;
};

#pragma region function definition

#pragma region member functions
template<typename T>
void LogisticRegression<T>::train(const Mat<T>& x, const Mat<string>& y)
{
    if (y.size_column() != 1)
		throw invalid_argument("Error: Matrix y must be single-column matrices.");
	if (x.size_row() != y.size_row())
		throw invalid_argument("Error: Matrix x and y must have the same number of rows.");
	if (x.size_row() < 1)
		throw invalid_argument("Error: Matrix x must have at least one row.");
    Mat<string> labels = unique(y);
    if(labels.size() != 2)  
        throw invalid_argument("Error: The target matrix y must contain exactly two unique classes for binary classification.");

    labels.sort_column(0);
    this->record(LABELS,labels);
    Mat<T> mumerical_y(y.size_row(),y.size_column());
    for(size_t i = 0; i < y.size_row(); ++i)
        mumerical_y.iloc(i,0) = (labels.iloc(0,0) == y.iloc(i,0) ? 1 : 0);

    Mat<T> ones(x.size_row(),1);
    ones.fill(1);
    Mat<T> w = concat_horizontal(x,ones);
    Mat<T> thetas(1,x.size_column()+1);

    // start training
    for(size_t i = 0; i < iterations; ++i)
    {
    cout<<"Output of the training iteration : "<<i+1<<endl;
    // generate random numbers
    if(x.size_row() < batch_size)
         throw out_of_range("Error: Batch size is larger than the available rows.");
    set<size_t> randomNums;
    random_device rd;
    mt19937 gen(rd()); 
    uniform_int_distribution<> dis(0, x.size_row()-1);
    while (randomNums.size() < batch_size) 
        randomNums.insert(dis(gen)); 
    
    Mat<T> tmp_thetas(thetas);

    for(size_t i = 0; i < w.size_column(); ++i)
    {
        T tmp_theta_i = 0;
        for(auto& e:randomNums)
        {
            tmp_theta_i += learning_rate*((mumerical_y.iloc_row(e) - LogisticRegression<T>::predict_probabilities(x.iloc_row(e),thetas)) * w.iloc(e,i)).iloc(0,0);
        }
        tmp_thetas.iloc(0,i) += (tmp_theta_i);
    }
    thetas = tmp_thetas;
    display_rainbow(thetas);
    }
    this->record(THETAS,thetas);
    
}
template<typename T> 
Mat<string> LogisticRegression<T>::predict(const Mat<T>& x) const
{   
    Mat<T> probabilities = predict_probabilities(x);
    Mat<string> ret(x.size_row(),1);
    for(size_t i = 0; i < x.size_row(); ++i)
        ret.iloc(i,0) = probabilities.iloc(i,0)>0.5?LABELS.read().iloc(0,0):LABELS.read().iloc(0,1);
    return ret;
}
template<typename T> 
Mat<T> LogisticRegression<T>::predict_probabilities(const Mat<T>& x) const
{
    return LogisticRegression<T>::predict_probabilities(x,THETAS);
}
#pragma endregion

#pragma region static functions
template<typename T>
Mat<T> LogisticRegression<T>::predict_probabilities(const Mat<T>& x,const Mat<T>& thetas)
{
	if (x.size_column()+1 != thetas.size_column())
         throw invalid_argument("Error: Number of columns in x must be equal to number of columns in thetas minus one.");

    Mat<T> y(x.size_row(),1);
    Mat<T> ones(x.size_row(),1);
    ones.fill(1);
    Mat<T> w = concat_horizontal(x,ones);
	w.dot(transpose(thetas));
    for(size_t i =0; i < x.size_row(); ++i)
        y.iloc(i,0) = 1 / (1 + exp(-w.iloc(i,0)));
    return y;
}
#pragma endregion

#pragma endregion

#endif // LINEAR_MODEL_H