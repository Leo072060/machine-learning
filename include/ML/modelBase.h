#ifndef MODELBASE_H
#define MODELBASE_H

#include"kits/managed.h"
#include"mat/mat.h"

using namespace std;







template<class T>
class RegressionModelBase:public ManagedClass
{
protected:
	RegressionModelBase(){};

// * * * * * * * functions * * * * * * *
public:
	virtual void    train  (const Mat<T>& x, const Mat<T>& y) = 0;
	virtual Mat<T>  predict(const Mat<T>& x) const			  = 0;
    virtual Dict<T> get_trainedParameters()  const            = 0;

// * * * * * * * attributes * * * * * * *
};







template<class T>
class ClassificationModelBase : public ManagedClass
{
protected:
	ClassificationModelBase(){};

// * * * * * * * functions * * * * * * *
public:
	virtual void		train  (const Mat<T>& x, const Mat<string>& y) = 0;
	virtual Mat<string> predict(const Mat<T>& x) const			       = 0;

// * * * * * * * attributes * * * * * * *
};





#endif // MODELBASE_H
