#ifndef MAT_H
#define MAT_H

#include<iostream>
#include<stdexcept>
#include<iomanip>
#include<functional>
#include<type_traits>
#include<math.h>
#include<set>

#include"kits/dict.h"
#include"kits/managed.h"

using namespace std;







#pragma region macro definition
using MATRIX_TYPE = const int;
#define MATRIX_TYPE_IDENTITY 1
using WITH_WHICH_NAME = uint8_t;
#define WITHOUT_NAME 0b11111111
#define WITH_ROWNAME 0b11111110
#define WITH_COLNAME 0b11111101
#define WITH_NAME    0b11111100
using ORDER = int;
#define ASCE 0
#define DESC 1
#pragma endregion

#pragma region forward declaration
template<class T> class Mat;

#pragma region non-member functions
template<typename T> Mat<T>       operator+        (const T& lhs, const Mat<T>& mat);
template<typename T> Mat<T>       operator-        (const T& lhs, const Mat<T>& mat);
template<typename T> Mat<T>       operator*        (const T& lhs, const Mat<T>& mat);
template<typename T> Mat<T>       operator/        (const T& lhs, const Mat<T>& mat);

template<typename T> Mat<T>       transpose        (const Mat<T>& mat);
template<typename T> Mat<T>       dot              (const Mat<T>& lhs, const Mat<T>& rhs);
template<typename T> Dict<Mat<T>> LU               (const Mat<T>& mat);
template<typename T> T            det              (const Mat<T>& mat);
template<typename T> T            det_LU           (const Mat<T>& mat);
template<typename T> T            det_inversion    (const Mat<T>& mat);
template<typename T> Mat<T>       inv              (const Mat<T>& mat);
template<typename T> Mat<T>       inv_Gauss_Jordan (const Mat<T>& mat);
template<typename T> T            mean             (const Mat<T>& mat);
template<typename T> Mat<T>       mean_row         (const Mat<T>& mat);
template<typename T> Mat<T>       mean_column      (const Mat<T>& mat);
template<typename T> T            sum              (const Mat<T>& mat);
template<typename T> Mat<T>       sum_row          (const Mat<T>& mat);
template<typename T> Mat<T>       sum_column       (const Mat<T>& mat);
template<typename T> Mat<T>       power            (const Mat<T>& mat,int exponent);
template<typename T> Mat<T>       abs              (const Mat<T>& mat);

template<typename T> Mat<T>       concat_horizontal(const Mat<T>& left, const Mat<T>& right);
template<typename T> Mat<T>       concat_vertical  (const Mat<T>& top, const Mat<T>& bottom);

template<typename T> Mat<T>       unique           (const Mat<T>& mat);

template<typename T> void         display          (const Mat<T>& mat, WITH_WHICH_NAME withWhichName = WITHOUT_NAME);
template<typename T> void         display_rainbow  (const Mat<T>& mat, WITH_WHICH_NAME withWhichName = WITHOUT_NAME);

// for matrix P
namespace P
{
	template<typename T> size_t CountSwaps(const Mat<T>& P);
}
#pragma endregion

#pragma endregion

template<class T = double>
class Mat : public ManagedClass
{
public:
	Mat ();
	Mat (const Mat<T>& other);
	Mat (const size_t rowSize, const size_t colSize);
	Mat (const size_t rowSize, const size_t colSize, MATRIX_TYPE matrixType);
	~Mat();

// * * * * * * * functions * * * * * * * 
	void		  operator=	       (const Mat<T>& other);
    void          operator+=       (const T& rhs);
    void          operator-=       (const T& rhs);
    void          operator*=       (const T& rhs);
    void          operator/=       (const T& rhs);
    Mat<T>        operator+        (const T& rhs)                                 const;
    Mat<T>        operator-        (const T& rhs)                                 const;
    Mat<T>        operator*        (const T& rhs)                                 const;                                 
    Mat<T>        operator/        (const T& rhs)                                 const;
    void          operator+=       (const Mat<T>& rhs)                            const; 
    void          operator-=       (const Mat<T>& rhs)                            const; 
    void          operator*=       (const Mat<T>& rhs)                            const;
    void          operator/=       (const Mat<T>& rhs)                            const; 
    Mat<T>        operator+        (const Mat<T>& rhs)                            const;
    Mat<T>        operator-        (const Mat<T>& rhs)                            const;
    Mat<T>        operator*        (const Mat<T>& rhs)                            const;                                 
    Mat<T>        operator/        (const Mat<T>& rhs)                            const;   
                    
	T& 			  iloc		       (const size_t i, const size_t j)					    { this->refresh(); return data[i][j]; }
	const T&	  iloc		       (const size_t i, const size_t j)               const { return data[i][j]; }
    Mat<T>        iloc_row         (const size_t i)                               const { return extract_rows(i,i+1); }
    Mat<T>        iloc_column      (const size_t i)                               const { return extract_columns(i,i+1); }
	T&			  loc			   (const string& rowName, const string& colName);
	const T&	  loc			   (const string& rowName, const string& colName) const;
	string&		  iloc_rowName     (const size_t& i)								    { return rowNames[i]; }
	const string& iloc_rowName     (const size_t& i)                              const { return rowNames[i]; }
	string&		  iloc_colName     (const size_t& i)								    { return colNames[i]; }
	const string& iloc_colName     (const size_t& i)                              const { return colNames[i]; }
	
    void          clear_names      ()                                                   { clear_rowNames(); clear_colNames(); }
	void          clear_rowNames   ();
	void          clear_colNames   ();

    size_t        size             ()                                             const { return rowSize*colSize; }
	size_t		  size_row	       ()                                             const { return rowSize; }
	size_t		  size_column	   ()                                             const { return colSize; }

	void		  swap_rows	       (const size_t a, const size_t b);
	void		  swap_columns     (const size_t a, const size_t b);

	Mat<T>        extract          (const size_t startRow, const size_t startCol, 
								    const size_t endRow,   const size_t endCol)   const;
	Mat<T>		  extract_rows     (const size_t startRow, const size_t endRow)	  const { return extract(startRow, 0, endRow, colSize); }
	Mat<T>		  extract_columns  (const size_t startCol, const size_t endCol)	  const { return extract(0, startCol, rowSize, endCol); }
    Mat<T>        extract_rows     (const vector<size_t>& index)                  const;
	Mat<string>	  extract_rowNames (const size_t startRow, const size_t endRow)   const;
	Mat<string>	  extract_colNames (const size_t startCol, const size_t endCol)   const;
	Mat<string>	  extract_rowNames ()										      const { return extract_rowNames(0, rowSize); }
	Mat<string>	  extract_colNames ()                                             const { return extract_colNames(0, colSize); }

    void          drop_rows        (const size_t startRow, const size_t endRow);
    void          drop_columns     (const size_t startCol, const size_t endCol);
    void          drop_row         (const size_t i)                                     { drop_rows(i,i+1); };
    void          drop_column      (const size_t i)                                     { drop_column(i,i+1); }

	void		  transpose		   ();
	void		  dot              (const Mat<T>& other);
    void          fill             (const T& val);

	void          concat_horizontal(const Mat<T>& other);
	void          concat_vertical  (const Mat<T>& other);

    void          sort_row         (const size_t i, ORDER order = ASCE);
    void          sort_column      (const size_t i, ORDER order = ASCE);

    Mat<size_t>   find             (const T val)                                  const;
private:
	T*			  operator[]	   (const size_t i)		                                { return data[i]; }
	const T*const operator[]	   (const size_t i)						          const { return data[i]; }
	// friend functions:
	friend Dict<Mat<T>>	LU<T>			   (const Mat<T>& mat);
	friend T			det<T>			   (const Mat<T>& mat);
	friend T			det_LU<T>		   (const Mat<T>& mat);
	friend T			det_inversion<T>   (const Mat<T>& mat);
	friend Mat<T>		inv<T>			   (const Mat<T>& mat);
	friend Mat<T>		inv_Gauss_Jordan<T>(const Mat<T>& mat);
    friend T            mean<T>            (const Mat<T>& mat);
    friend Mat<T>       mean_row<T>        (const Mat<T>& mat);
    friend Mat<T>       mean_column<T>     (const Mat<T>& mat);
    friend T            sum<T>             (const Mat<T>& mat);
    friend Mat<T>       sum_row<T>         (const Mat<T>& mat);
    friend Mat<T>       sum_column<T>      (const Mat<T>& mat);
    
// * * * * * * * attributes * * * * * * *
private:
	T**					data		= nullptr;
	string*				rowNames	= nullptr;		
	string*				colNames	= nullptr;
	size_t				rowSize		= 0;
	size_t				colSize		= 0;
	// calculated value
	mutable ManagedVal<T>		DET;
    mutable ManagedVal<T>       MEAN;
    mutable ManagedVal<Mat<T>>  MEAN_ROW;
    mutable ManagedVal<Mat<T>>  MEAN_COLUMN;
    mutable ManagedVal<T>       SUM;
    mutable ManagedVal<Mat<T>>  SUM_ROW;
    mutable ManagedVal<Mat<T>>  SUM_COLUMN;
};

#pragma region function definition

#pragma region lifecycle management
template<typename T>
Mat<T>::Mat():ManagedClass(),
              DET(this->administrator), 
              MEAN(this->administrator),MEAN_ROW(this->administrator),MEAN_COLUMN(this->administrator),
              SUM(this->administrator),SUM_ROW(this->administrator),SUM_COLUMN(this->administrator) {}
template<typename T>
Mat<T>::Mat(const Mat<T>& other) :Mat() 
{
	if (0 == other.rowSize || 0 == other.rowSize) return;
	rowSize	 = other.size_row();
	colSize	 = other.size_column();
	data	 = new T * [rowSize];
	rowNames = new string[rowSize];
	colNames = new string[colSize];
	for (size_t i = 0; i < rowSize; ++i)
	{
		data[i] = new T[colSize];
		rowNames[i] = other.rowNames[i];
		for (size_t j = 0; j < colSize; ++j)
			data[i][j] = other[i][j];
	}
	for (size_t i = 0; i < colSize; ++i) 
		colNames[i] = other.colNames[i];
}
template<typename T>
Mat<T>::Mat(const size_t rowSize,const size_t colSize) :Mat() 
{
	this->rowSize =	rowSize;
	this->colSize =	colSize;
	data		  =	new T * [rowSize];
	rowNames	  =	new string[rowSize];
	colNames	  =	new string[colSize];
	for (size_t i = 0; i < rowSize; ++i)
	{
		data[i] = new T[colSize];
		for (size_t j= 0; j < colSize; ++j) 
			data[i][j] = T();
	}
}
template<typename T>
Mat<T>::Mat(const size_t rowSize,const size_t colSize,MATRIX_TYPE matrixType) :Mat(rowSize,colSize)
{
	switch (matrixType)
	{
	case MATRIX_TYPE_IDENTITY:
		if(rowSize!=colSize) 
			throw invalid_argument("Error: Identity matrix must be square (rowSize must equal colSize).");
		for (size_t i = 0; i < rowSize; ++i) 
			data[i][i] = 1;
		break;
	default:
		break;
	}
}
template<typename T>
Mat<T>::~Mat()
{
	if (data)
		for (size_t i = 0; i < rowSize; ++i) 
			delete[] data[i];
	if (rowNames) delete[] rowNames;
	if (colNames) delete[] colNames;
}
#pragma endregion

#pragma region member functions
template<typename T>
void Mat<T>::operator=(const Mat<T>& other) {
	for (size_t i=0; i < rowSize; ++i) 
		delete[] data[i];
	rowSize	= other.size_row();
	colSize	= other.size_column();
	data	= new T * [rowSize];
	for (size_t i = 0; i < rowSize; ++i)
	{
		data[i] = new T[colSize];
		for (size_t j = 0; j < colSize; ++j) 
			data[i][j] = other[i][j];
	}
}
template<typename T>
void Mat<T>::operator+=(const T& rhs)
{
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j = 0; j < colSize; ++j)
            data[i][j]+=rhs;
    this->refresh();
}
template<typename T>
void Mat<T>::operator-=(const T& rhs)
{
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j = 0; j < colSize; ++j)
            data[i][j]-=rhs;
    this->refresh();
}
template<typename T>
void Mat<T>::operator*=(const T& rhs)
{
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j = 0; j < colSize; ++j)
            data[i][j]*=rhs;
    this->refresh();
}
template<typename T>
void Mat<T>::operator/=(const T& rhs)
{
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j = 0; j < colSize; ++j)
            data[i][j]/=rhs;
    this->refresh();
}
template<typename T>
Mat<T> Mat<T>::operator+(const T& rhs) const
{
    Mat<T> lhs(*this);
    lhs += rhs;
    return lhs;
}
template<typename T>
Mat<T> Mat<T>::operator-(const T& rhs) const
{
    Mat<T> lhs(*this);
    lhs -= rhs;
    return lhs;
}
template<typename T>
Mat<T> Mat<T>::operator*(const T& rhs) const 
{
    Mat<T> lhs(*this);
    lhs *= rhs;
    return lhs;
}
template<typename T>                              
Mat<T> Mat<T>::operator/(const T& rhs) const 
{
    Mat<T> lhs(*this);
    lhs /= rhs;
    return lhs;
}
template<typename T>
void Mat<T>::operator+=(const Mat<T>& rhs) const
{
    if(rowSize!=rhs.size_row() || colSize!=rhs.size_column())
      throw invalid_argument("Error: Matrix dimensions must match for this operation.");
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j =0; j < colSize; ++j)
            data[i][j] += rhs.iloc(i,j);
}
template<typename T>
void Mat<T>::operator-= (const Mat<T>& rhs) const
{
    if(rowSize != rhs.size_row() || colSize != rhs.size_column())
        throw invalid_argument("Error: Matrix dimensions must match for this operation.");
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j =0; j < colSize; ++j)
            data[i][j] -= rhs.iloc(i,j);
}
template<typename T>
void Mat<T>::operator*= (const Mat<T>& rhs) const
{
    if(rowSize!=rhs.size_row() || colSize!=rhs.size_column())
      throw invalid_argument("Error: Matrix dimensions must match for this operation.");
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j =0; j < colSize; ++j)
            data[i][j] *= rhs.iloc(i,j);
}
template<typename T>
void Mat<T>::operator/= (const Mat<T>& rhs) const
{
    if(rowSize!=rhs.size_row() || colSize!=rhs.size_column())
      throw invalid_argument("Error: Matrix dimensions must match for this operation.");
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j =0; j < colSize; ++j)
            data[i][j] /= rhs.iloc(i,j);
}
template<typename T>
Mat<T> Mat<T>::operator+(const Mat<T>& rhs) const
{
    Mat<T> ret(*this);
    ret += rhs;
    return ret;
}
template<typename T>
Mat<T> Mat<T>::operator-(const Mat<T>& rhs) const
{
    Mat<T> ret(*this);
    ret -= rhs;
    return ret;
}
template<typename T>
Mat<T> Mat<T>::operator*(const Mat<T>& rhs) const
{
    Mat<T> ret(*this);
    ret *= rhs;
    return ret;
}                             
template<typename T>
Mat<T> Mat<T>::operator/(const Mat<T>& rhs) const
{
    Mat<T> ret(*this);
    ret /= rhs;
    return ret;
}
template<typename T>
T& Mat<T>::loc(const string& rowName, const string& colName)
{
	size_t i = 0, j = 0;
	for (; i < rowSize; ++i)
		if (rowNames[i] == rowName) 
			break;
	if(i==rowSize)
		throw invalid_argument("Error: Row name '" + rowName + "' not found.");
	for (; j < colSize; ++j) 
		if (colNames[j] == colName) 
			break;
	if(j==colSize)	
		throw invalid_argument("Error: Column name '" + colName + "' not found.");
	this->refresh();
	return data[i][j];
}
template<typename T>
const T& Mat<T>::loc(const string& rowName, const string& colName) const
{
	size_t i = 0, j = 0;
	for (; i < rowSize; ++i)
		if (rowNames[i] == rowName)
			break;
	if (i == rowSize)
		throw invalid_argument("Error: Row name '" + rowName + "' not found.");
	for (; j < colSize; ++j)
		if (colNames[j] == colName)
			break;
	if (j == colSize)
		throw invalid_argument("Error: Column name '" + colName + "' not found.");
	return data[i][j];
}
template<typename T>
void Mat<T>::clear_rowNames()
{
	for (size_t i = 0; i < rowSize; ++i)
		rowNames[i].clear();
	this->refresh();
}
template<typename T>
void Mat<T>::clear_colNames()
{
	for (size_t i = 0; i < colSize; ++i)
		colNames[i].clear();
	this->refresh();
}
template<typename T>
void Mat<T>::swap_rows(const size_t a,const size_t b)
{
	if (a >= rowSize || b >= rowSize) 
		throw invalid_argument("Error: Row index out of bounds.");
	if (a == b) return;
	swap(data[a], data[b]);
	swap(rowNames[a], rowNames[b]);
	this->refresh();
}
template<typename T>
void Mat<T>::swap_columns(const size_t a,const size_t b)
{
	if (a >= colSize || b >= colSize)
		throw invalid_argument("Error: Row index out of bounds.");
	if (a == b) return;
	for (size_t i = 0; i < rowSize; ++i) 
		swap(data[a][i],data[b][i]);
	swap(colNames[a], colNames[b]);
	this->refresh();
}
template<typename T>
Mat<T> Mat<T>::extract(const size_t startRow, const size_t startCol, const size_t endRow, const size_t endCol) const
{
	if(startRow>=endRow || startCol>=endCol)
		throw invalid_argument("Error: Start index must be less than end index.");
	if(endRow>rowSize || endCol>colSize)
		throw out_of_range("Error: End row or column index exceed matrix dimensions.");
	Mat<T> mat(endRow - startRow, endCol - startCol);
	for (size_t i = 0; i < endRow - startRow; ++i)
		for (size_t j = 0; j < endCol - startCol; ++j)
			mat.data[i][j] = data[i + startRow][j + startCol];
	for (size_t i = 0; i < endRow - startRow; ++i)
		mat.rowNames[i] = rowNames[i + startRow];
	for (size_t j = 0; j < endCol - startCol; ++j)
		mat.colNames[j] = colNames[j + startCol];
	return mat;
}
template<typename T>
Mat<T> Mat<T>::extract_rows(const vector<size_t>& index) const
{
    Mat<T> ret(index.size(),colSize);
    int i = 0;
    for(const auto& e:index)
        for(size_t j=0; j<colSize;++j)
            ret.data[i][j]=data[e][j];
    return ret;
}
template<typename T>
Mat<string>	Mat<T>::extract_rowNames(const size_t startRow, const size_t endRow) const
{
	if (startRow >= endRow)
		throw invalid_argument("Error: Start index must be less than end index.");
	if (endRow > rowSize)
		throw out_of_range("Error: End row or column index exceed matrix dimensions.");
	Mat<string> mat(endRow - startRow, 1);
	for (size_t i = 0; i < (endRow - startRow); ++i)
		mat.iloc(i,0) = rowNames[startRow + i];
	return mat;
}
template<typename T>
Mat<string>	Mat<T>::extract_colNames(const size_t startCol, const size_t endCol) const
{
	if (startCol >= endCol)
		throw invalid_argument("Error: Start index must be less than end index.");
	if (endCol > colSize)
		throw out_of_range("Error: End row or column index exceed matrix dimensions.");
	Mat<string> mat(1, endCol - startCol);
	for (size_t i = 0; i < (endCol - startCol); ++i)
		mat.iloc(0,i) = colNames[startCol + i];
	return mat;
}
template<typename T>
void Mat<T>::drop_rows(const size_t startRow, const size_t endRow)
{
    if(startRow >= rowSize || startRow < 0)
        throw out_of_range("Error: Start row index is out of bounds.");
    if(endRow > rowSize || endRow < 0)
        throw out_of_range("Error: End row index is out of bounds.");
    if(startRow == endRow) return;
    size_t new_rowSize   = rowSize-endRow+startRow;
    T** new_data         = new T*[new_rowSize];
    string* new_rowNames = new string[new_rowSize];
    for(size_t i = 0; i < startRow; ++i)
    {
        new_data[i]     = data[i];
        data[i]         = nullptr;
        new_rowNames[i] = rowNames[i];
    }
    for(size_t i = startRow; i < endRow; ++i)
    {
        delete data[i];
    }
    for(size_t i = endRow; i < rowSize; ++i)
    {
        new_data[i-endRow+startRow]     = data[i];
        data[i]                         = nullptr;
        new_rowNames[i-endRow+startRow] = rowNames[i];
    }
    delete[] data;
    delete[] rowNames;
    data     = new_data;
    rowNames = new_rowNames;
    rowSize  = new_rowSize;
    this->refresh();
}
template<typename T>
void Mat<T>::drop_columns(const size_t startCol, const size_t endCol)
{
    if (startCol >= colSize || startCol < 0)
        throw out_of_range("Error: Start column index is out of bounds.");
    if (endCol > colSize || endCol < 0)
        throw out_of_range("Error: End column index is out of bounds.");
    if (startCol == endCol) return;
    size_t new_colSize = colSize - endCol + startCol;
    T** new_data         = new T*[rowSize];
    string* new_colNames = new string*[new_colSize];
    for(size_t i = 0; i < rowSize; ++i)
    {
        new_data[i] = new T[new_colSize];
        for(size_t j = 0; j < startCol; ++j)
            new_data[i][j] = data[i][j];
        for(size_t j = endCol; j < colSize; ++j)
            new_data[i][j-endCol+startCol] = data[i][j];
        delete[] data[i];
    }
    for(size_t j = 0; j < startCol; ++j)
        new_colNames[j] = colNames[j];
    for(size_t j = endCol; j < colSize; ++j)
        new_colNames[j-endCol+startCol] = colNames[j];
    delete[] colNames;
    colNames = new_colNames;
    colSize  = new_colSize;
    this->refresh();
}
template<typename T>
void Mat<T>::transpose()
{
	T** new_data = new T * [colSize];
	for (size_t i = 0; i < colSize; ++i)
	{
		new_data[i] = new T[rowSize];
		for (size_t j = 0; j < rowSize; ++j)
			new_data[i][j] = data[j][i];
	}

    for (size_t i = 0; i < rowSize; i++)
        delete[] data[i];
	delete[] data;
	data = new_data;
	swap(rowSize, colSize);
	swap(rowNames, colNames);
	this->refresh();
}
template<typename T>
void Mat<T>::dot(const Mat<T>& other)
{
	if (colSize != other.rowSize)
		throw invalid_argument("Matrix dimensions do not align for dot product: " +
								to_string(colSize) + " (cols of A) != " +
								to_string(other.rowSize) + " (rows of B)");
	T** new_data = new T * [rowSize];
	for (size_t i = 0; i < rowSize; ++i)
	{
		new_data[i] = new T[other.colSize];
		for (size_t j = 0; j < other.colSize; ++j)
		{
			T sum = 0;
			for (size_t k = 0; k < colSize; ++k)
				sum += data[i][k] * other.data[k][j];
			new_data[i][j] = sum;
		}
	}
	delete[] colNames;
	colNames = new string[other.colSize];
	for (size_t i = 0; i < other.colSize; ++i)
		colNames[i] = other.colNames[i];
	for (size_t i = 0; i < rowSize; ++i)
		delete[] data[i];
	data = new_data;
	colSize = other.colSize;
	this->refresh();
}
template<typename T>
void Mat<T>::fill(const T& val)
{
    for(size_t i = 0; i < rowSize; ++i)
            for(size_t j = 0; j < colSize; ++j)
                data[i][j] = val;
}
template<typename T>
void Mat<T>::concat_horizontal(const Mat<T>& other)
{
    if (rowSize != other.rowSize)
        throw invalid_argument("Row sizes must be equal for horizontal concatenation.");
	T**     new_data     = new T * [rowSize];
	size_t  new_colSize  = colSize + other.colSize;
	string* new_colNames = new string[new_colSize];
	for (size_t i = 0; i < rowSize; ++i)
	{
		new_data[i] = new T[new_colSize];
		for (size_t j = 0; j < colSize; ++j)
			new_data[i][j] = data[i][j];
		for (size_t j = 0; j < other.colSize; ++j)
			new_data[i][j + colSize] = other.data[i][j];
		delete[] data[i];
	}
	delete[] data;
	data = new_data;
	for (size_t i = 0; i < colSize; ++i)
		new_colNames[i] = colNames[i];
	for (size_t i = 0; i < other.colSize; ++i)
		new_colNames[i + colSize] = other.colNames[i];
	delete[] colNames;
	colNames = new_colNames;
	colSize  = new_colSize;
}
template<typename T>
void Mat<T>::concat_vertical(const Mat<T>& other)
{
	if (colSize != other.colSize)
		throw invalid_argument("Column sizes must be equal for vertical concatenation.");
	T** new_data = new T * [rowSize + other.rowSize];
	size_t new_rowSize = rowSize + other.rowSize;
	string* new_rowNames = new string[new_rowSize];
	for (size_t i = 0; i < rowSize; ++i)
	{
		new_data[i] = new T[colSize];
		new_data[i] = data[i];
		new_rowNames[i] = rowNames[i];
	}
	for (size_t i = 0; i < other.rowSize; ++i)
	{
		new_data[i + rowSize] = new T[colSize];
		for (size_t j = 0; j < colSize; ++j)
			new_data[i + rowSize][j] = other.data[i][j];
		new_rowNames[i+rowSize] = other.rowNames[i];
	}
	delete[] data;
	data     = new_data; 
	delete[] rowNames;
    rowNames = new_rowNames;
	rowSize  = new_rowSize; 
}
template<typename T>
void Mat<T>::sort_row(const size_t i, const ORDER order)
{
    if(i >= rowSize) throw out_of_range("Error: Row index " + to_string(i) + " is out of range. Maximum row index is " + to_string(rowSize - 1) + ".");
    function<bool(const size_t,const size_t)> cmp;
    if     (order == ASCE) cmp = [this,i](const size_t lhs, const size_t rhs) { return data[i][lhs] < data[i][rhs]; };
    else if(order == DESC) cmp = [this,i](const size_t lhs, const size_t rhs) { return data[i][lhs] > data[i][rhs]; };
    else throw invalid_argument("Error: Invalid order argument. Expected ASCE or DESC.");
 
    this->transpose();
    this->sort_column(i,order);
    this->transpose();
}
template<typename T>
void Mat<T>::sort_column(const size_t i, const ORDER order)
{
    if(i >= colSize) throw out_of_range("Error: Column index " + to_string(i) + " is out of range. Maximum column index is " + to_string(colSize - 1) + ".");
    function<bool(const size_t,const size_t)> cmp;
    if     (order == ASCE) cmp = [this,i](const size_t lhs, const size_t rhs) { return data[lhs][i] < data[rhs][i]; };
    else if(order == DESC) cmp = [this,i](const size_t lhs, const size_t rhs) { return data[lhs][i] > data[rhs][i]; };
    else throw invalid_argument("Error: Invalid order argument. Expected ASCE or DESC.");

    vector<size_t> index(rowSize); for(size_t i = 0; i < rowSize; ++i) index[i] = i;
    sort(index.begin(),index.end(),cmp);
    
    T** new_data = new T*[rowSize];
    string* new_rowNames = new string[rowSize];
    for(size_t i = 0; i < rowSize; ++i) 
    {
        new_data[i]     = data[index[i]];     data[index[i]] = nullptr;
        new_rowNames[i] = rowNames[index[i]]; 
    }
    delete[] data;     data = new_data;
    delete[] rowNames; rowNames = new_rowNames;
}
template<typename T> 
Mat<size_t> Mat<T>::find(const T val) const
{
    for(size_t i = 0; i <rowSize; ++i)
        for(size_t j = 0; j < colSize; ++j)
        {
            if(data[i][j] == val)
            {
                Mat<size_t> ret(1,2);
                ret.iloc(0,0) = i;
                ret.iloc(0,1) = j;
                return ret;
            }
        }
    Mat<size_t> ret(0,0);
    return ret;
}   
#pragma endregion

#pragma region non-member functions

template<typename T> 
Mat<T> operator+(const T& lhs, const Mat<T>& mat) { return mat + lhs; }
template<typename T>
Mat<T> operator-(const T& lhs, const Mat<T>& mat) { return mat - lhs; }
template<typename T>
Mat<T> operator*(const T& lhs, const Mat<T>& mat) { return mat * lhs; }
template<typename T>
Mat<T> operator/(const T& lhs, const Mat<T>& mat) { return mat / lhs; }
template<typename T>
Mat<T> transpose(const Mat<T>& mat)
{
	Mat<T> ret(mat.size_column(), mat.size_row());
	for (size_t i = 0; i < ret.size_row(); ++i)
		for (size_t j = 0; j < ret.size_column(); ++j)
			ret.iloc(i, j) = mat.iloc(j, i);
	for (size_t i = 0; i < ret.size_row(); ++i)
		ret.iloc_rowName(i) = mat.iloc_colName(i);
	for (size_t i = 0; i < ret.size_column(); ++i)
		ret.iloc_colName(i) = mat.iloc_rowName(i);
	return ret;
}
template<typename T>
Mat<T> dot(const Mat<T>& lhs, const Mat<T>& rhs)
{
	Mat<T> ret(lhs);
	ret.dot(rhs);
    return ret;
}
template<typename T>
Dict<Mat<T>> LU(const Mat<T>& mat) 
{
	if (mat.size_row() != mat.size_column())
		throw invalid_argument("Error: Matrix must be square to compute determinant.");
	if (mat.size_row() <= 1)
		throw invalid_argument("Error: Matrix must have more than one row to perform LU decomposition.");
	Mat<T>	L(mat.size_row(), mat.size_column()),
		    U(mat),
		    P(mat.size_row(), mat.size_column(), MATRIX_TYPE_IDENTITY);
	U.clear_names();
	size_t	d		   = 0; // d is for depth
	size_t	countSwaps = 0;
	while (true)
	{
		if (d == mat.size_row() - 1)
		{
			L[d][d] = U[d][d] / U[d][d];
			break;
		}
		// partial pivoting
		T maxVal = fabs(U[d][d]);
		size_t maxRow = d;
		for (size_t i = d + 1; i < mat.size_row(); ++i)
		{
			if (fabs(U[i][d]) > maxVal)
			{
				maxVal = fabs(U[i][d]);
				maxRow = i;
			}
		}
		if (maxRow != d)
		{
			U.swap_rows(d, maxRow);
			P.swap_rows(d, maxRow);
			++countSwaps;
		}
		if (fabs(U[d][d]) < 1e-9)
			throw runtime_error("Error: Matrix is singular and cannot be decomposed.");
		for (size_t i = d; i < mat.size_row(); ++i)	
			L[i][d] = U[i][d] / U[d][d];
		for (size_t i = d + 1; i < mat.size_row(); ++i)	
			U[i][d] = 0;
		for (size_t i = d + 1; i < mat.size_row(); ++i)
			for (size_t j = d + 1; j < mat.size_column(); ++j)
				U[i][j] -= L[i][d] * U[d][j];
		++d;
	}
	Dict<Mat<T>> ret;
	ret.insert("L", L);
	ret.insert("U", U);
	ret.insert("P", P);
	return ret;
}
template<typename T>
T det(const Mat<T>& mat) 
{
	return det_LU(mat); 
}
template<typename T>
T det_LU(const Mat<T>& mat)
{
	if (mat.DET.readable()) 
		return mat.DET.read();
	if (mat.size_row() != mat.size_column())
		throw invalid_argument("Error: Matrix must be square to compute determinant.");
	T		det = 1;
	auto	ret = LU(mat);
	auto	U   = ret["U"];
	for (size_t i = 0; i < mat.size_row(); ++i) 
		det *= U[i][i];
	if (P::CountSwaps(ret["P"]) % 2) 
		return -det;
	mat.record(mat.DET, det);
	return mat.DET.read();
}
template<typename T>
T det_inversion(const Mat<T>& mat) 
{
	if (mat.DET.readable()) 
		return mat.DET.read();
	if (mat.size_row() != mat.size_column()) 
		throw invalid_argument("Error: Matrix must be square to compute determinant.");
	T det = 0;
	bool* usedCol = new bool[mat.size_column()];
	for (size_t i = 0; i < mat.size_column(); ++i) 
		usedCol[i] = false;
	// define a Lambda expression for recursion
	function<void(size_t, T, T&, bool*, size_t)> _det_inversion =
		[&_det_inversion, &mat](size_t rowIndex, T term, T& det, bool* usedCol, size_t inversionCount)
		{
			if (rowIndex >= mat.size_row())
			{
				if (inversionCount % 2) 
					term *= -1;
				det += term;
				return;
			}
			for (size_t colIndex = 0; colIndex < mat.size_column(); ++colIndex)
			{
				if (usedCol[colIndex] || !mat.iloc(rowIndex,colIndex)) continue;
				size_t new_inversionCount = inversionCount;
				for (size_t i = mat.size_column() - 1; i > colIndex; --i)
					if (usedCol[i]) 
						++new_inversionCount;
				usedCol[colIndex] = true;
				_det_inversion(rowIndex + 1, term * mat.iloc(rowIndex,colIndex), det, usedCol, new_inversionCount);
				usedCol[colIndex] = false;
			}
		};

	det_recursive(0, 1, det, usedCol, 0);
	delete[] usedCol;
	mat.record(mat.administrator, mat.DET, det);
	return mat.DET.read();
}
template<typename T>
Mat<T> inv(const Mat<T>& mat)
{
	return inv_Gauss_Jordan(mat);
}
template<typename T>
Mat<T> inv_Gauss_Jordan(const Mat<T>& mat) 
{
	if (mat.size_row() != mat.size_column())
		throw invalid_argument("Error: Matrix must be square to compute inverse.");
	Mat<T> A(mat);
	Mat<T> I(mat.size_row(), mat.size_column(), MATRIX_TYPE_IDENTITY);
	for (size_t d = 0; d < mat.size_row(); ++d) // d is for depth
	{
		// partial pivoting
		T      maxVal    = fabs(A[d][d]);
		size_t maxValRow = d;
		for (size_t i = d + 1; i < mat.size_row(); ++i)
		{
			if (fabs(A[i][d]) > maxVal) 
			{
				maxVal    = fabs(A[i][d]);
				maxValRow = i;
			}
		}
		if (maxValRow != d)
		{
			A.swap_rows(d, maxValRow);
			I.swap_rows(d, maxValRow);
		}
		if (fabs(A[d][d]) < 1e-9) 
			throw runtime_error("Error: Matrix is singular and cannot be inverted.");
		
		T pivot = A[d][d];
		for (size_t j = 0; j < mat.size_row(); ++j)
		{
			A[d][j] /= pivot;
			I[d][j] /= pivot;
		}
		for (size_t i = 0; i < mat.size_row(); ++i)
		{
			if (i != d)
			{
				T factor = A[i][d];
				for (size_t j = 0; j < mat.size_row(); ++j)
				{
					A[i][j] -= factor * A[d][j];
					I[i][j] -= factor * I[d][j];
				}
			}
		}
	}
	return I;
}
template<typename T> 
T mean (const Mat<T>& mat)
{
    if(mat.MEAN.readable()) return mat.MEAN.read();
    mat.record(mat.MEAN,mean_column(mean_row(mat)).iloc(0,0));
    return mat.MEAN.read();
}
template<typename T> 
Mat<T> mean_row(const Mat<T>& mat)
{
    if(mat.MEAN_ROW.readable()) return mat.MEAN_ROW.read();
    Mat<T> ret(mat.size_row(),1);
    for(size_t i = 0; i < mat.size_row(); ++i)
    {
        T mean = 0;
        for(size_t j = 0; j < mat.size_column(); ++j)
        {
            mean += (mat.iloc(i,j)-mean)/(j+1); 
        }
        ret.iloc(i,0) = mean;
    }
    mat.record(mat.MEAN_ROW,ret);
    return mat.MEAN_ROW.read();
}
template<typename T> 
Mat<T> mean_column(const Mat<T>& mat)
{
    if(mat.MEAN_COLUMN.readable()) return mat.MEAN_COLUMN.read();
    Mat<T> ret(1,mat.size_column());
    for(size_t i = 0; i < mat.size_column(); ++i)
    {
        T mean = 0;
        for(size_t j = 0; j < mat.size_row(); ++j)
        {
            mean += (mat.iloc(j,i)-mean)/(j+1);
        }
        ret.iloc(0,i) = mean;
    }
    mat.record(mat.MEAN_COLUMN,ret);
    return mat.MEAN_COLUMN.read();
}
template<typename T> 
T sum(const Mat<T>& mat)
{
    if(mat.SUM.readable()) return mat.SUM.read();
    mat.record(mat.SUM,sum_column(sum_row(mat)).iloc(0,0));
    return mat.SUM.read();
}
template<typename T> 
Mat<T> sum_row(const Mat<T>& mat)
{
    if(mat.SUM_ROW.readable()) return mat.SUM_ROW.read();
    Mat<T> ret(mat.size_row(),1);
    for(size_t i = 0; i < mat.size_row(); ++i)
        for(size_t j = 0; j < mat.size_column(); ++j)
            ret.iloc(i,0) += mat.iloc(i,j);
    mat.record(mat.SUM_ROW,ret);
    return mat.SUM_ROW.read();
}
template<typename T> 
Mat<T> sum_column(const Mat<T>& mat)
{
    if(mat.SUM_COLUMN.readable()) return mat.SUM_COLUMN.read();
    Mat<T> ret(1,mat.size_column());
    for(size_t i = 0; i < mat.size_column(); ++i)
        for(size_t j = 0; j < mat.size_row(); ++j)
            ret.iloc(0,i) += mat.iloc(j,i);
    mat.record(mat.SUM_COLUMN,ret);
    return mat.SUM_COLUMN.read();
}
template<typename T> 
Mat<T> power(const Mat<T>& mat,int exponent)
{
    Mat<T> ret(mat.size_row(),mat.size_column());
    for(size_t i = 0; i < mat.size_row(); ++i)
        for(size_t j = 0; j < mat.size_column(); ++j)
            ret.iloc(i,j) = pow(mat.iloc(i,j),exponent);
    return ret;
}
template<typename T> 
Mat<T> abs(const Mat<T>& mat)
{
    Mat<T> ret(mat.size_row(),mat.size_column());
    for(size_t i = 0; i < mat.size_row(); ++i)
        for(size_t j = 0; j < mat.size_column(); ++j)
            ret.iloc(i,j) = abs(mat.iloc(i,j));
    return ret;
}
template<typename T> 
Mat<T> concat_horizontal(const Mat<T>& left, const Mat<T>& right)
{
    Mat<T> ret(left);
    ret.concat_horizontal(right);
    return ret;
}
template<typename T> 
Mat<T>& concat_vertical(const Mat<T>& top, const Mat<T>& bottom)
{
    Mat<T> ret(top);
    ret.concat_vertical(bottom);
    return ret;
}

template<typename T> 
Mat<T> unique(const Mat<T>& mat)
{
    set<T> S;
    for(size_t i = 0; i < mat.size_row(); ++i)
        for(size_t j = 0; j < mat.size_column(); ++j)
            S.insert(mat.iloc(i,j));
    Mat<T> ret(1,S.size());
    size_t i = 0;
    for(auto &e : S)
        ret.iloc(0,i++) = e;
    return ret;
}

template<typename T>
void display(const Mat<T>& mat, WITH_WHICH_NAME withWhichName)
{
	const int width_val     = 15;
	const int width_rowName = 10;
	const int precision     = 5;
	cout << "Shape: " << mat.size_row() << " * " << mat.size_column() << endl;
	if (CHECK_FLAG(withWhichName, WITH_COLNAME))
	{
		if (CHECK_FLAG(withWhichName, WITH_ROWNAME))
			cout << fixed << setw(width_rowName) << " ";
		for (size_t i = 0; i < mat.size_column(); ++i)
			cout << fixed << setw(width_val) << mat.iloc_colName(i);
		cout << endl;
	}
	for (size_t i = 0; i < mat.size_row(); ++i)
	{
		if (CHECK_FLAG(withWhichName, WITH_ROWNAME))
			cout << fixed << setw(width_rowName) << mat.iloc_rowName(i);
		for (size_t j = 0; j < mat.size_column(); ++j)
		{
			const T val = mat.iloc(i,j);
			if constexpr (is_same<T, string>::value)
				cout << setw(width_val) << val;
			else if constexpr (is_floating_point<T>::value)
			{
				if (val == static_cast<int>(val))
					cout << fixed << setw(width_val) << static_cast<int>(val);
				else
					cout << fixed << setprecision(precision) << setw(width_val) << val;
			}
			else cout << setw(width_val) << val;
		}
		cout << endl;
	}
}
// use ANSI escape codes for font color
template<typename T>
void display_rainbow(const Mat<T>& mat, WITH_WHICH_NAME withWhichName)
{
	const int width_val     = 15;
	const int width_rowName = 10;
	const int precision     = 5;
	cout << "Shape: " << mat.size_row() << " * " << mat.size_column() << endl;
	if (CHECK_FLAG(withWhichName, WITH_COLNAME))
	{
		if (CHECK_FLAG(withWhichName, WITH_ROWNAME))
			cout << fixed << setw(width_rowName) << " ";
		for (size_t i = 0; i < mat.size_column(); ++i)
		{
			cout << "\033[" << (i % 2 == 0 ? "32m" : "34m"); // alternate text color
			cout << fixed << setw(width_val) << mat.iloc_colName(i);
		}
		cout << "\033[0m" << endl; // reset colors
	}
	for (size_t i = 0; i < mat.size_row(); ++i) {
		cout << "\033[" << (i % 2 == 0 ? "48;5;235" : "48;5;240") << "m"; // alternate background color

		if (CHECK_FLAG(withWhichName, WITH_ROWNAME))
			cout << fixed << setw(width_rowName) << mat.iloc_rowName(i);
		for (size_t j = 0; j < mat.size_column(); ++j)
		{
			cout << "\033[" << (j % 2 == 0 ? "32m" : "34m"); // alternate text color
			const T  val = mat.iloc(i, j);
			if constexpr (is_same<T, string>::value)
				cout << setw(width_val) << val;
			else if constexpr (is_floating_point<T>::value)
			{
				if (val == static_cast<int>(val))
					cout << fixed << setw(width_val) << static_cast<int>(val);
				else
					cout << fixed << setprecision(precision) << setw(width_val) << val;
			}
			else cout << setw(width_val) << val;
		}
		cout << "\033[0m" << endl; // reset colors
	}
}
template<typename T>
size_t P::CountSwaps(const Mat<T>& P)
	{
		size_t swapsCount = 0;
		// verify if it is a P matrix 
		// and record the column positions where each row has a value of 1
		vector<size_t> rec(P.size_row());
		for (size_t i = 0; i < P.size_row(); ++i)
		{
			bool find_1 = false;
			for (size_t j = 0; j < P.size_column(); ++j)
			{
				if (1 == P.iloc(i,j))
				{
					if (find_1 == false)
					{
						rec[i] = j;
						find_1    = true;
					}
					else  
						throw invalid_argument("Error: Row " + to_string(i) + " has multiple leading 1s in the permutation matrix.");
				}
			}
			if (find_1 == false)  
				throw invalid_argument("Error: Row " + to_string(i) + " does not have a leading 1 in the permutation matrix.");
		}

		for (size_t i = 0; i < P.size_row(); ++i)
		{
			if (rec[i] != i)
			{
				size_t j = i + 1;
				for (; j < P.size_row(); ++j)
				{
					if (rec[j] == i)
					{
						rec[j] = rec[i];
						++swapsCount;
						break;
					}
				}
				if (j == P.size_row())
					throw invalid_argument("Error: Row " + to_string(i) + " does not have a leading 1 in the permutation matrix.");
			}
		}
		return swapsCount;
	}

#pragma endregion

#pragma endregion

#endif // MAT_H
