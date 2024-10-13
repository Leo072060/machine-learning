#ifndef DICT_H
#define DICT_H

#include<map>
#include<string>

using namespace std;







template<class T>
class Dict
{
// * * * * * * * functions * * * * * * *
public:
	void insert		(string key, T value);
	T	 operator[]	(string key) {return data[key];}

// * * * * * * * attributes * * * * * * *
private:
	map<string, T> data;
};

#pragma region function definition

#pragma region member functions
template<typename T>
void Dict<T>::insert(string key, T value)
{
	data.insert({ key, value });
}
#pragma endregion

#pragma endregion







#endif // DICT_H
