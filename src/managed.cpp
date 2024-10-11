#include"kits/managed.h"

using namespace std;

ADMINISTRATOR_ID Administrator::GlobalID = 0;

void SetPermission(const Administrator& admin, ManagedBase& managedBase, const PERMISSION perm)
{
	if (admin.ID != managedBase.administrator_ID)
		throw runtime_error("Error: This administrator does not have permission to modify.");
	managedBase.permission = perm;
}

void AddPermission(const Administrator& administrator, ManagedBase& managedBase, const PERMISSION perm)
{
	if (managedBase.check_permission(perm)) return;
	managedBase.permission = managedBase.permission & perm;
}