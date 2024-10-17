#include<stdexcept>

#include"mat/mat.h"
#include"kits/dict.h"

using namespace std;

template<typename T>
class Scaler
{
public:
    Scaler():x_max(administrator),x_min(administrator){};
    void fit(const Mat<T>& mat);

private:
    Administrator administrator;
public:
    ManagedVal<Mat<T>> x_max;
    ManagedVal<Mat<T>> x_min;
};


#pragma function definition

#pragma member function

#pragma endregion

#pragma endregion
template<typename T>
Dict<Mat<T>> min_max_normalization(const Mat<T>& x)
{
    if (x.size_column() == 0 || x.size_row() == 0)
        throw runtime_error("Error: Input matrix x cannot be empty.");
    T* x_max = new T[x.size_column()];
    T* x_min = new T[x.size_column()];
    Mat<T> scaling(1,x.size_column());
    Mat<T> scaled_x(x.size_row(),x.size_column());
    for (size_t i = 0; i < x.size_column(); ++i)
    {
        x_max[i] = x.iloc(0, i); 
        x_min[i] = x.iloc(0, i); 
        for (size_t j = 0; j < x.size_row(); ++j)
        {
            if (x.iloc(j, i) > x_max[i])
                x_max[i] = x.iloc(j, i);
            if (x.iloc(j, i) < x_min[i])
                x_min[i] = x.iloc(j, i);
        }
        scaling.iloc(0,i) = x_max[i] - x_min[i];
        if (scaling.iloc(0,i) == 0) 
            scaling.iloc(0,i) = 1;
    }
    for (size_t i = 0; i < x.size_column(); ++i)
        for (size_t j = 0; j < x.size_row(); ++j)
             scaled_x.iloc(j, i) = (x.iloc(j, i) - x_min[i]) / scaling.iloc(0, i);
    Dict<Mat<T>> ret;
    ret.insert("scaled_x",scaled_x);
    ret.insert("scaling",scaling);
    delete[] x_max;
    delete[] x_min;
    return ret;
}
