#include"kits/managed.h"

using namespace std;

ADMINISTRATOR_ID Administrator::GlobalID = 0;
void Administrator::registerManagedItem(const ManagedItem& managedItem,const PERMISSION& perm) const
{
    SetPermission(*this,managedItem,perm);
    managedItemList.push_back(&managedItem);
}
void Administrator::setManagedItemsPermission(const PERMISSION& perm) const
{
    for(const auto& e : managedItemList)
        SetPermission(*this,*e,perm);
}
void Administrator::addManagedItemsPermission(const PERMISSION& perm) const
{
    for(const auto& e : managedItemList)
        AddPermission(*this,*e,perm);    
}







void ManagedItem::setPermission(const Administrator& admin, const PERMISSION& perm) const
{
	if (admin.ID != this->administrator_ID)
		throw runtime_error("Error: This administrator does not have permission to modify.");
	permission = perm;
}
void ManagedItem::addPermission(const Administrator& admin, const PERMISSION& perm) const
{
    if (admin.ID != this->administrator_ID)
		throw runtime_error("Error: This administrator does not have permission to modify.");
	if (this->checkPermission(perm)) return;
	permission = permission & perm;
}
void SetPermission(const Administrator& admin, const ManagedItem& managedItem, const PERMISSION& perm)
{
	managedItem.setPermission(admin, perm);
}
void AddPermission(const Administrator& admin, const ManagedItem& managedItem, const PERMISSION& perm)
{
	managedItem.addPermission(admin, perm);
}