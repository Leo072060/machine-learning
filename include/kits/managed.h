#ifndef MANAGED_H
#define MANAGED_H

#include<stdexcept>
#include<memory>
#include<vector>

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







#pragma region forward declaration

class Administrator;
class ManagedItem;
void SetPermission(const Administrator& admin,
				   const ManagedItem& managedItem,
				   const PERMISSION& perm);
void AddPermission(const Administrator& admin,
				   const ManagedItem& managedItem,
				   const PERMISSION& perm);

#pragma endregion

class Administrator
{
public:
	Administrator() :ID(GlobalID++) {}

// * * * * * * * functions * * * * * * *
    void registerManagedItem(const ManagedItem& managedItem,const PERMISSION& perm = PERMISSION_LOWEST) const;
    void setManagedItemsPermission(const PERMISSION& perm) const;
    void addManagedItemsPermission(const PERMISSION& perm) const;

// * * * * * * * attributes * * * * * * *
public:
	const  ADMINISTRATOR_ID	ID;
private:
	static ADMINISTRATOR_ID	GlobalID;
    mutable vector<const ManagedItem*> managedItemList;
};

#pragma region function defination

#pragma region member functions



#pragma endregion

#pragma endregion







class ManagedItem
{
protected:
	ManagedItem(const Administrator& admin,const PERMISSION& perm = PERMISSION_LOWEST)
		:administrator_ID(admin.ID) { admin.registerManagedItem(*this,perm); }
    ManagedItem(const Administrator& admin,const ManagedItem& other)
        :ManagedItem(admin,other.permission) { } 

// * * * * * * * functions * * * * * * *
public:
	bool	   checkPermission(const PERMISSION& perm) const { return CHECK_FLAG(permission,perm); }
	PERMISSION getPermission  ()                       const { return permission; }
	bool	   readable		   ()                      const { return checkPermission(PERMISSION_READ); }
public:
    void       setPermission(const Administrator& admin, const PERMISSION& perm) const;
    void       addPermission(const Administrator& admin, const PERMISSION& perm) const;

// * * * * * * * attributes * * * * * * *
public:
	const ADMINISTRATOR_ID administrator_ID;
protected:
	mutable PERMISSION     permission;
};

#pragma region function defination

#pragma member functions



#pragma endregion

#pragma endregion






#pragma region forward declaration

template<class T> class ManagedVal;

#pragma region non-member functions

template<typename T> void Copy(const Administrator& admin, ManagedVal<T>& managedVal, const ManagedVal<T>& other);

#pragma endregion

#pragma endregion

template<class T>
class ManagedVal :public ManagedItem
{
public:
	ManagedVal(const Administrator& admin, const PERMISSION& perm = PERMISSION_LOWEST)
        :ManagedItem(admin, perm), value(nullptr) {};
    ManagedVal(const Administrator& admin, const ManagedVal& other)
        :ManagedItem(other),value(*other.value){};

// * * * * * * * functions * * * * * * *
public:
	operator T	     () const {return static_cast<T>(*value); }
	const T&	read () const;
	void		write(const T& val);
	// friend functions:
	friend void	Copy<>(const Administrator& admin, ManagedVal<T>& managedVal, const ManagedVal<T>& other);

// * * * * * * * attributes * * * * * * *
private:
	unique_ptr<T> value;
};

#pragma region function definition

#pragma region member functions
template<typename T>
const T& ManagedVal<T>::read() const
{
	if (!checkPermission(PERMISSION_READ))
		throw runtime_error("Error: Insufficient permissions for read operation.");
	return *value;
}
template<typename T>
void ManagedVal<T>::write(const T& val)
{
	if (!checkPermission(PERMISSION_WRITE))
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







template<typename T>
class ManagedClass
{
protected:
    ManagedClass();

// * * * * * * * functions * * * * * * *
protected:
template<typename Y> void record (ManagedVal<Y>& managedVal, const Y& val) const;
                     void refresh()                                        const;
private:
    Administrator administrator;
    mutable bool isrefreshed = true;
};

#pragma region function declaration

#pragma region member functions

template<typename T>
template<typename Y>
void ManagedClass<T>::record(ManagedVal<Y>& managedVal, const Y& val) const
{
	SetPermission(administrator, managedVal, PERMISSION_WRITE);
    managedVal.write(val);
    SetPermission(administrator, managedVal, PERMISSION_READ);
	isrefreshed = false;
}
template<typename T>
void ManagedClass<T>::refresh() const
{
    if(isrefreshed) return;
    administrator.setManagedItemsPermission(PERMISSION_LOWEST);
    isrefreshed = true;
}

#pragma endregion

#pragma endregion

#endif // MANAGED_H
