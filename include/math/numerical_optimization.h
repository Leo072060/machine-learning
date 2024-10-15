#include<random>

#include"mat/mat.h"

using namespace std;

#pragma function declaration
template<typename T> Mat<T> gradient_descent(const Mat<T>& x, const Mat<T>& y, const size_t batch_size, const T learning_rate, const size_t iterations);
template<typename T> Mat<T> newton_method   (const Mat<T>& x, const Mat<T>& y, const size_t batch_size, const size_t iterations);
#pragma endregion

#pragma function difination
template<typename T>
Mat<T> gradient_descent(const Mat<T>& x, const Mat<T>& y, const size_t batch_size, const T learning_rate, const size_t iterations)
{
    Mat<T> ones(x.size_row(),1);
    for(size_t i = 0; i < x.size_row(); ++i) ones.iloc(i,0) = 1;
    Mat<T> w = concat_horizontal(x,ones);
    Mat<T> thetas(1,x.size_column()+1);
    // start training
    for(size_t i = 0; i < iterations; ++i)
    {
#ifdef DEBUG
    cout<<"iteration : "<< i + 1 <<endl;
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
    return thetas;
}
template<typename T>
Mat<T> newton_method(const Mat<T>& x,const Mat<T>& y,size_t batch_size,size_t iterations)
{
    Mat<T> ret;
    return ret;
}
#pragma endregion