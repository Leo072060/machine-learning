#ifndef MANAGED_H
#define MANAGED_H

#include<stdexcept>
#include<memory>
#include<vector>

using namespace std;

#pragma region macro definition
#define ADMINISTRATOR_ID  unsigned long 
// permission
#define PERMISSION		  uint8_t
#define PERMISSION_LOWEST 0b11111111
#define PERMISSION_READ	  0b11111110
#define PERMISSION_WRITE  0b11111101
#define CHECK_FLAG(NAME, FLAG) ((NAME | FLAG) == FLAG)
#pragma endregion







#pragma region forward declaration
class ManagedItem;
#pragma endregion

class Administrator
{
public:
	Administrator() :ID(GlobalID++) {}

// * * * * * * * functions * * * * * * *
    void               registerConstManagedItem  (const ManagedItem& managedItem) const;
    void               registerManagedItem       (ManagedItem& managedItem)       const;
    size_t             constManagedItemsSize     ()                               const { return constManagedItemList.size(); }
    size_t             managedItemsSize          ()                               const { return managedItemList.size(); }
    const ManagedItem& getConstManagedItem       (const size_t i)                 const { return *constManagedItemList[i]; }
    ManagedItem&       getManagedItem            (const size_t i)                 const { return *managedItemList[i]; }

// * * * * * * * attributes * * * * * * *
public:
	const  ADMINISTRATOR_ID	ID;
private:
    mutable vector<unique_ptr<const ManagedItem>> constManagedItemList;
    mutable vector<unique_ptr<ManagedItem>>       managedItemList;
    static ADMINISTRATOR_ID	GlobalID; // a counter to assigned number
};









class ManagedItem
{
protected:
	ManagedItem(const Administrator& admin)
        :administrator_ID(admin.ID) { admin.registerManagedItem(*this); }

// * * * * * * * functions * * * * * * *
public:
	bool	   checkPermission(const PERMISSION perm)                             const { return CHECK_FLAG(permission,perm); }
	PERMISSION getPermission  ()                                                  const { return permission; }
	bool	   readable		  ()                                                  const { return checkPermission(PERMISSION_READ); }
    void       setPermission  (const Administrator& admin, const PERMISSION perm) const;
    void       addPermission  (const Administrator& admin, const PERMISSION perm) const;
    
// * * * * * * * attributes * * * * * * *
public:
	const ADMINISTRATOR_ID administrator_ID;
protected:
	mutable PERMISSION     permission = PERMISSION_LOWEST;
};







template<class T>
class ManagedVal :public ManagedItem
{
public:
	ManagedVal(const Administrator& admin)
        :ManagedItem(admin), value(nullptr) {};
    ManagedVal(const Administrator& admin, const T& val)
        :ManagedVal(admin) { value = make_unique<T>(val); }

// * * * * * * * functions * * * * * * *
public:
    operator const T&(){ return read(); }
	const T& read () const;
	void     write(const T& val);

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

#pragma endregion







class ManagedClass
{
protected:
    ManagedClass() = default;

// * * * * * * * functions * * * * * * *
protected:                    
template<typename Y> 
void record         (ManagedVal<Y>& managedVal, const Y& val) const;
void refresh        ()                                        const;

// * * * * * * * attributes * * * * * * *
protected:
const Administrator administrator;
mutable bool isrefreshed = true;
};

#pragma region function declaration

#pragma region member functions
template<typename Y>
void ManagedClass::record(ManagedVal<Y>& managedVal, const Y& val) const
{
	managedVal.setPermission(administrator, PERMISSION_WRITE);
    managedVal.write(val);
    managedVal.setPermission(administrator, PERMISSION_READ);
	isrefreshed = false;
}
#pragma endregion

#pragma endregion

#endif // MANAGED_H
