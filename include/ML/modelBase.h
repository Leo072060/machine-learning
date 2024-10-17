#ifndef MODELBASE_H
#define MODELBASE_H

#include"kits/managed.h"
#include"mat/mat.h"

using namespace std;








template<class T>
class RegressionModelBase:public ManagedClass
{
protected:
	RegressionModelBase(){}
    RegressionModelBase(const RegressionModelBase<T>& other){}
// * * * * * * * functions * * * * * * *
public:
	virtual void    train  (const Mat<T>& x, const Mat<T>& y) = 0;
	virtual Mat<T>  predict(const Mat<T>& x) const			  = 0;
    virtual shared_ptr<RegressionModelBase<T>>  clone() const = 0;
// * * * * * * * attributes * * * * * * *

};







template<class T>
class ClassificationModelBase : public ManagedClass
{
protected:
	ClassificationModelBase(){};
    ClassificationModelBase(const ClassificationModelBase& other){}
// * * * * * * * functions * * * * * * *
public:
	virtual void		train  (const Mat<T>& x, const Mat<string>& y) = 0;
	virtual Mat<string> predict(const Mat<T>& x) const			       = 0;
    virtual shared_ptr<ClassificationModelBase>  clone() const = 0;
// * * * * * * * attributes * * * * * * *

};






// #pragma region forward declaration
// using MULTI_CLASS_TYPE = int;
// #define OvO 1
// #define OvR 2
// #define MvM 3
// #pragma endregion

// template<class T>
// class BinaryClassificationModel : public ManagedClass
// {
// protected:
//     BinaryClassificationModel(){};
// // * * * * * * * functions * * * * * * *
// public:
//     void train

// // * * * * * * * attributes * * * * * * *
// public:
//     MULTI_CLASS_TYPE multi_class = OvR;
// }

#endif // MODELBASE_H
