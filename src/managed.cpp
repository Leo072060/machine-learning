#include"kits/managed.h"

using namespace std;






#pragma region class Administrator
ADMINISTRATOR_ID Administrator::GlobalID = 0;

#pragma region member functions
void Administrator::registerConstManagedItem(const ManagedItem& managedItem) const
{
    constManagedItemList.push_back(make_unique<const ManagedItem>(managedItem));
}
void Administrator::registerManagedItem(ManagedItem& managedItem) const
{
    managedItemList.push_back(make_unique<ManagedItem>(managedItem));
}
#pragma endregion

#pragma endregion







#pragma region class  ManagedItem

#pragma region functions

#pragma region member functions
void ManagedItem::setPermission(const Administrator& admin, const PERMISSION perm) const
{
	if (admin.ID != this->administrator_ID)
		throw runtime_error("Error: This administrator does not have permission to modify.");
	permission = perm;
}
void ManagedItem::addPermission(const Administrator& admin, const PERMISSION perm) const
{
    if (admin.ID != this->administrator_ID)
		throw runtime_error("Error: This administrator does not have permission to modify.");
	if (this->checkPermission(perm)) return;
	permission = permission & perm;
}
#pragma endregion

#pragma endregion

#pragma endregion







#pragma region class ManagedClass

#pragma region function declaration

#pragma region member functions
void ManagedClass::refresh() const
{
    if(isrefreshed) return;
    for(size_t i = 0; i < administrator.constManagedItemsSize(); ++i)
    {
        administrator.getConstManagedItem(i).setPermission(administrator,PERMISSION_LOWEST);
    }
    for(size_t i = 0; i < administrator.managedItemsSize(); ++i)
    {
        administrator.getManagedItem(i).setPermission(administrator,PERMISSION_LOWEST);
    }
    isrefreshed = true;
}
#pragma endregion

#pragma endregion

#pragma endregion