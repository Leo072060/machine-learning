#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include<fstream>
#include<sstream>
#include<vector>

#include"mat/mat.h"

using namespace std;







#pragma region forward declaration

#pragma region non-member functions
template<typename T> T str2T(const string& str);
#pragma endregion

#pragma endregion

template<class T>
class Loader
{
protected:
    Loader() {}

    // * * * * * * * functions * * * * * * *
public:
    virtual Mat<T> load_matrix(const string& fileName) const = 0;

    // * * * * * * * attributes * * * * * * *
public:
};

template<class T>
class csv_Loader:public Loader<T>
{
public:
    csv_Loader() {}

// * * * * * * * functions * * * * * * *
public:
    Mat<T> load_matrix(const string& fileName) const override;

// * * * * * * * attributes * * * * * * *
public:
    WITH_WHICH_NAME with_which_name = WITHOUT_NAME;
};

#pragma region function definition
#pragma region member functions
template<typename T>
Mat<T> csv_Loader<T>::load_matrix(const string& fileName) const
{
    ifstream file(fileName);
    if (!file.is_open())
        throw runtime_error("Error: Could not open file: " + fileName);
    vector<vector<T>>lines;
    string           line;
    vector<string>   colNames;
    vector<string>   rowNames;
    size_t           rowSize = 0;
    size_t           columnSize=0;
    // process the first row
    if (getline(file, line))
    {
        istringstream headerStream(line);
        string        headerCell;
        vector<T>     line_data;
        //  process the first cell
        getline(headerStream, headerCell, ',');
        // special handling for UTF-8 encoding to accommodate character data correctly
        if (headerCell.size() >= 3 
            && headerCell[0] == (char)0xEF
            && headerCell[1] == (char)0xBB
            && headerCell[2] == (char)0xBF)
        {
            headerCell = headerCell.substr(3);
        }
        if      (with_which_name == WITH_NAME);
        else if (with_which_name == WITH_COLNAME) colNames.push_back(headerCell);
        else if (with_which_name == WITH_ROWNAME) rowNames.push_back(headerCell);
        else                                      line_data.push_back(str2T<T>(headerCell));

        while (getline(headerStream, headerCell, ','))
        {
            if   (CHECK_FLAG(with_which_name, WITH_COLNAME)) colNames.push_back(headerCell);
            else                                             line_data.push_back(str2T<T>(headerCell));
        }
        if (!CHECK_FLAG(with_which_name, WITH_COLNAME))
        {
            lines.push_back(line_data);
            ++rowSize;
        }
        columnSize = CHECK_FLAG(with_which_name, WITH_COLNAME) ? colNames.size() : line_data.size();
    }
    else return Mat<T>();
    // process other rows
    while (getline(file, line))
    {
        istringstream ss(line);
        string        cell;
        vector<T>     line_data;
        getline(ss, cell, ',');
        if (CHECK_FLAG(with_which_name,WITH_ROWNAME)) rowNames.push_back(cell);
        else                                          line_data.push_back(str2T<T>(cell));
        while (getline(ss, cell, ','))
        {
            line_data.push_back(str2T<T>(cell));
        }
        if (line_data.size() != columnSize )
            throw runtime_error("Error: Row data size does not match the expected column size at row " + to_string(rowSize + 1));
        lines.push_back(line_data);
        ++rowSize;
    }
    // store in the matrix
    Mat<T> mat(rowSize, columnSize);
    for (size_t i = 0; i < mat.size_row(); ++i)
        for (size_t j = 0; j < mat.size_column(); ++j)
            mat.iloc(i, j) = lines[i][j];
    if (CHECK_FLAG(with_which_name, WITH_ROWNAME))
        for (size_t i = 0; i < mat.size_row(); ++i)
            mat.iloc_rowName(i) = rowNames[i];
    if (CHECK_FLAG(with_which_name, WITH_COLNAME))
        for (size_t i = 0; i < mat.size_column(); ++i)
            mat.iloc_colName(i) = colNames[i];
    
    file.close();

    return mat;
}
#pragma endregion

#pragma region non-member functions

template<typename T>
T str2T(const string& str)
{
    istringstream iss(str);
    T val;
    iss >> val;
    if (iss.fail())
        throw std::invalid_argument("Error: Invalid value: " + str);
    return val;
}

#pragma endregion

#pragma endregion

#endif //DATA_LOADER_H