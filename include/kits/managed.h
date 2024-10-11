#ifndef MANAGED_H
#define MANAGED_H

#include<stdexcept>
#include<memory>

using namespace std;

#pragma region macro definition

#define ADMINISTRATOR_ID  uint8_t
#define PERMISSION		  uint8_t
#define PERMISSION_LOWEST 0b11111111
#define PERMISSION_READ	  0b11111110
#define PERMISSION_WRITE  0b11111101
/*
0	PERMISSION_LOWEST
1	PERMISSION_READ		read
2	PERMISSION_WRITE	read	write
*/
#define CHECK_FLAG(NAME, FLAG) ((NAME | FLAG) == FLAG)

#pragma endregion







class Administrator
{
public:
	Administrator() :ID(GlobalID++) {}
    
// * * * * * * * attributes * * * * * * *
public:
	const  ADMINISTRATOR_ID	ID;
private:
	static ADMINISTRATOR_ID	GlobalID;
};








class ManagedBase
{
protected:
	ManagedBase(const Administrator& admin,const PERMISSION perm = PERMISSION_LOWEST)
		:administrator_ID(admin.ID), permission(perm) {}
    ManagedBase(const Administrator& admin,const ManagedBase& other)
        :administrator_ID(other.administrator_ID),permission(other.permission){}
public:
	virtual ~ManagedBase() = default;

// * * * * * * * functions * * * * * * *
public:
	bool	   check_permission(const PERMISSION perm) const { return CHECK_FLAG(permission,perm); }
	PERMISSION get_permission  ()                      const { return permission; }
	bool	   readable		   ()                      const { return check_permission(PERMISSION_READ); }
	// friend functions:
	friend void	SetPermission (const Administrator& admin,
							   ManagedBase& managedBase,
							   const PERMISSION perm);
	friend void AddPermission (const Administrator& admin,
							   ManagedBase& managedBase,
							   const PERMISSION perm);


// * * * * * * * attributes * * * * * * *
public:
	const ADMINISTRATOR_ID administrator_ID;
protected:
	PERMISSION			   permission;
};







#pragma region forward declaration

template<class T> class ManagedVal;

#pragma region non-member functions

template<typename T> void ForceWrite(const Administrator& admin, ManagedVal<T>& managedVal, const T& value);
template<typename T> void Copy      (const Administrator& admin, ManagedVal<T>& managedVal, const ManagedVal<T>& other);

#pragma endregion

#pragma endregion

template<class T>
class ManagedVal :public ManagedBase
{
public:
	ManagedVal(const Administrator& admin, const PERMISSION perm = PERMISSION_LOWEST)
        :ManagedBase(admin, perm), value(nullptr) {};
    ManagedVal(const Administrator& admin, const ManagedVal& other)
        :ManagedBase(other),value(*other.value){};
	// ManagedVal(ManagedVal&& other) noexcept
    //     :ManagedBase(move(other)),value(move(other.value)){}
// * * * * * * * functions * * * * * * *
public:
	operator T	     () const {return static_cast<T>(*value); }
	const T&	read () const;
	void		write(const T& val);
	// friend functions:
	friend void	Copy<>		(const Administrator& admin, ManagedVal<T>& managedVal, const ManagedVal<T>& other);

// * * * * * * * attributes * * * * * * *
private:
	unique_ptr<T> value;
};

#pragma region function definition

#pragma region member functions
template<typename T>
const T& ManagedVal<T>::read() const
{
	if (!check_permission(PERMISSION_READ))
		throw runtime_error("Error: Insufficient permissions for read operation.");
	return *value;
}
template<typename T>
void ManagedVal<T>::write(const T& val)
{
	if (!check_permission(PERMISSION_WRITE))
		throw runtime_error("Error: Insufficient permissions for write operation.");
	value = make_unique<T>(val);
}
#pragma endregion

#pragma region non-member functions

template<typename T>
void Copy(const Administrator& admin, ManagedVal<T>& managedVal, const ManagedVal<T>& other)
{
	if (admin.ID != managedVal.administrator_ID)
		throw runtime_error("Error: This administrator does not have permission to modify this value.");
	if (other.value == nullptr) managedVal.value = nullptr;
	else managedVal.value = make_unique<T>(*other.value);
	managedVal.permission = other.permission;
}

#pragma endregion

#pragma endregion







#endif // MANAGED_H
