#ifndef LINEAR_MODEL_H
#define LINEAR_MODEL_H

#include<stdexcept> 
#include<vector>
#include<random>
#include<set>

#include "modelBase.h"
#include"managed.h"
#include"numatrix.h"
#include"dict.h"







template<typename T = double>
class SimpleLinearRegression:public RegressionModelBase
{
public:
	SimpleLinearRegression() :W(administrator), B(administrator), M(administrator) {}

// * * * * * * * functions * * * * * * * 
public:
	void    train                (const Mat<T>& x, const Mat<T>& y)              override; 
	Mat<T>  predict              (const Mat<T>& x)                         const override;
	T       predict              (const T& x)	                           const;
	Dict<T> get_trainedParameters();

// * * * * * * * attributes * * * * * * *
private:
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

	record(M, x.size_row())

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

	record(W, w);
	record(B, b);
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
template<typename T>
Dict<T> SimpleLinearRegression<T>::get_trainedParameters()
{
	Dict<T> ret;
	ret.insert("w", W.read());
	ret.insert("b", B.read());
	return ret;
}
#pragma endregion 
#pragma endregion 







template<typename T = double>
class LinearRegression :public RegressionModelBase
{

// * * * * * * * functions * * * * * * *
public:
	void    train(const Mat<T>& x, const Mat<T>& y) override;
	Mat<T>  predict(const Mat<T>& x) const			override;

// * * * * * * * attributes * * * * * * *
public:
	// model parameters
	double learning_rate;
	size_t batch_size;
	size_t iterations;
private:
	// calculated value
	ManagedVal<vector<T>> THETAS;
};

#pragma region function definition
#pragma region member functions
template<typename T>
void LinearRegression<T>::train(const Mat<T>& x, const Mat<T>& y)
{
	if (y.size_column() != 1)
		throw invalid_argument("Error: Matrix y must be single-column matrices.");
	if (x.size_row() != y.size_row())
		throw invalid_argument("Error: Matrix x and y must have the same number of rows.");
	if (x.size_row() < 1)
		throw invalid_argument("Error: Matrix x must have at least one row.");

    Mat<T> ones(x.size_row(),1);
    for(size_t i = 0; i < x.size_row(); ++i) ones.iloc(i,0) = 1;
    Mat<T> w = concat_horizontal(x,ones);
    Mat<T> thetas(1,x.size_column());

    // 
    for(size_t i = 0; i < iterations; ++i)
    {
    // generate random numbers
    set<size_t> randomNums;
    random_device rd;
    mt19937 gen(rd()); 
    uniform_int_distribution<> dis(0, x.size_row()-1);
    while (randomNums.size() < batch_size) 
        randomNums.insert(dis(gen)); 
    
    Mat<T> tmp_thetas(thetas);

    for(size_t i = 0; i < x.size_column(); ++i)
    {
        tmp_theta_i = 0;
        for(auto& e:randomNums)
        {
            tmp_theta_i += learning_rate*((y.iloc_row(e) - det(x.iloc_row(e),transpose(theta))) * x.iloc(e,i)).iloc(0,0);
        }
        tmp_theta.iloc(i) += (tmp_theta_i/batch_size);
    }
    thetas = tmp_theta;
    }
    record(THETAS,thetas);
}
template<typename T>
Mat<T> LinearRegression<T>::predict(const Mat<T>& x) const
{
	if (x.size_column() != 1)
		throw invalid_argument("Error: Input matrix x must be a single-column matrix for prediction.");
	Mat<T> y(1, x.size_row);
	for (size_t i = 0; i < x.size_row(); ++i)
		y.iloc(i, 0) = x.iloc(i, 0) * W.read() + B.read();
	return y;
}
#pragma endregion 
#pragma endregion 





#endif // LINEAR_MODEL_H