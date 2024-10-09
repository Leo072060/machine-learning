#ifndef MODELBASE_H
#define MODELBASE_H

#include"mat.h"

using namespace std;







template<class T>
class RegressionModelBase
{
private:
	RegressionModelBase();

// * * * * * * * functions * * * * * * *
public:
	virtual void   train  (const Mat<T>& x, const Mat<T>& y) = 0;
	virtual Mat<T> predict(const Mat<T>& x) const			 = 0;
private:
	template<typename Y>
	void           record (ManagedVal<Y>& managedVal, const Y& val) const;
    
// * * * * * * * attributes * * * * * * *
protected:
	mutable bool		isRefreshed = true;
	const Administrator administor;
};

#pragma region function definition
#pragma region member functions
template<typename T> template<typename Y>
void RegressionModelBase<T>::record(ManagedVal<Y>& managedVal, const Y& val) const
{
	ForceWrite<Y>(administrator, managedVal, val);
	SetPermission(administrator, managedVal, PERMISSION_READ);
	isRefreshed = false;
}
#pragma endregion
#pragma endregion





template<class T>
class ClassificationModelBase
{
private:
	ClassificationModelBase();

// * * * * * * * functions * * * * * * *
public:
	virtual void		train  (const Mat<T>& x, const Mat<string>& y) = 0;
	virtual Mat<string> predict(const Mat<T>& x) const			       = 0;
private:
	template<typename Y>
	void           record(ManagedVal<Y>& managedVal, const Y& val) const;
// * * * * * * * attributes * * * * * * *
protected:
	mutable bool		isRefreshed = true;
	const Administrator administor;
};

#pragma region function definition
#pragma region member functions
template<typename T> template<typename Y>
void ClassificationModelBase<T>::record(ManagedVal<Y>& managedVal, const Y& val) const
{
	ForceWrite<Y>(administrator, managedVal, val);
	SetPermission(administrator, managedVal, PERMISSION_READ);
	isRefreshed = false;
}
#pragma endregion
#pragma endregion





#endif // MODELBASE_H
