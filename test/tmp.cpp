#pragma region forward declaration
template<class T> class Mat;
#pragma region non-member functions
namespace P
{
	template<typename T> size_t CountSwaps(const Mat<T>& P);
}
template<typename T>      Mat<T>       transpose        (const Mat<T>& mat);
template<typename T>      Mat<T>       dot              (const Mat<T>& lhs, const Mat<T>& rhs);
template<typename T>      Mat<T>       concat_horizontal(const Mat<T>& left, const Mat<T>& right);
template<typename T>      Mat<T>       concat_vertical  (const Mat<T>& top, const Mat<T>& bottom);
template<typename T>      void         display          (const Mat<T> mat, WITH_WHICH_NAME withWhichName = WITHOUT_NAME);
template<typename T>      void         display_rainbow  (const Mat<T> mat, WITH_WHICH_NAME withWhichName = WITHOUT_NAME);
#pragma region friend functions     
template<typename T,typename U> Mat<T> operator+        (const U& lhs, const Mat<T>& mat);
template<typename T,typename U> Mat<T> operator-        (const U& lhs, const Mat<T>& mat);
template<typename T,typename U> Mat<T> operator*        (const U& lhs, const Mat<T>& mat);
template<typename T,typename U> Mat<T> operator/        (const U& lhs, const Mat<T>& mat);
template<typename T>      Dict<Mat<T>> LU               (const Mat<T>& mat);
template<typename T>      T            det              (const Mat<T>& mat);
template<typename T>      T            det_LU           (const Mat<T>& mat);
template<typename T>      T            det_inversion    (const Mat<T>& mat);
template<typename T>      Mat<T>       inv              (const Mat<T>& mat);
template<typename T>      Mat<T>       inv_Gauss_Jordan (const Mat<T>& mat);
#pragma endregion
#pragma endregion
#pragma endregion

template<class T=double>
class Mat
{
public:
	Mat ();
	Mat (const Mat<T>& other);
	// Mat (Mat&& other) noexcept;
	Mat (const size_t rowSize, const size_t colSize);
	Mat (const size_t rowSize, const size_t colSize, MATRIX_TYPE matrixType);
	~Mat();

// * * * * * * * functions * * * * * * * 
	void		  operator=	       (const Mat<T>& other);
	// void		  operator=        (Mat<T>&& other) noexcept;
    template<typename U> void   operator+= (const U& rhs);
    template<typename U> void   operator-= (const U& rhs);
    template<typename U> void   operator*= (const U& rhs);
    template<typename U> void   operator/= (const U& rhs);
    template<typename U> Mat<T> operator+  (const U& rhs)                         const;
    template<typename U> Mat<T> operator-  (const U& rhs)                         const;
    template<typename U> Mat<T> operator*  (const U& rhs)                         const;                                 
    template<typename U> Mat<T> operator/  (const U& rhs)                         const;                              
	
	// friend functions:
    template<typename U> friend Mat<T> operator+ (const U& lhs, const Mat<T>& mat);
    template<typename U> friend Mat<T> operator- (const U& lhs, const Mat<T>& mat);
    template<typename U> friend Mat<T> operator* (const U& lhs, const Mat<T>& mat);
    template<typename U> friend Mat<T> operator/ (const U& lhs, const Mat<T>& mat);
	friend Dict<Mat<T>>	LU<T>			   (const Mat<T>& mat);
	friend T			det<T>			   (const Mat<T>& mat);
	friend T			det_LU<T>		   (const Mat<T>& mat);
	friend T			det_inversion<T>   (const Mat<T>& mat);
	friend Mat<T>		inv<T>			   (const Mat<T>& mat);
	friend Mat<T>		inv_Gauss_Jordan<T>(const Mat<T>& mat);
private:
	T*			   operator[]		 (const size_t i)		                         { return data[i]; }
	const T* const operator[]		 (const size_t i)						   const { return data[i]; }
	template<typename Y>
	void		   record			 (ManagedVal<Y>& managedVal, const Y& val) const;
	void		   refresh			 ()                                        const;
	void		   copy_calculatedVal(const Mat<T>& other)                     const;

// * * * * * * * attributes * * * * * * *
private:
	T**					data		= nullptr;
	string*				rowNames	= nullptr;		
	string*				colNames	= nullptr;
	size_t				rowSize		= 0;
	size_t				colSize		= 0;
	mutable bool		isRefreshed = true;
	const Administrator	administrator;
	// calculated value
	mutable ManagedVal<T>		DET;
	mutable ManagedVal<Mat<T>>	L;
	mutable ManagedVal<Mat<T>>	U;
	mutable ManagedVal<Mat<T>>	P;
};

template<typename T> template<typename U>
void Mat<T>::operator+= (const U& rhs)
{
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j = 0; j < colSize; ++j)
            data[i][j]+=rhs;
}
template<typename T> template<typename U>
void Mat<T>::operator-= (const U& rhs)
{
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j = 0; j < colSize; ++j)
            data[i][j]-=rhs;
}
template<typename T> template<typename U>
void Mat<T>::operator*= (const U& rhs)
{
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j = 0; j < colSize; ++j)
            data[i][j]*=rhs;
}
template<typename T> template<typename U>
void Mat<T>::operator/= (const U& rhs)
{
    for(size_t i = 0; i < rowSize; ++i)
        for(size_t j = 0; j < colSize; ++j)
            data[i][j]/=rhs;
}
template<typename T> template<typename U>
Mat<T> Mat<T>::operator+ (const U& rhs) const
{
    Mat<T> lhs(*this);
    return lhs += rhs;
}
template<typename T> template<typename U>
Mat<T> Mat<T>::operator- (const T& rhs) const
{
    Mat<T> lhs(*this);
    return lhs -= rhs;
}
template<typename T> template<typename U>
Mat<T> Mat<T>::operator* (const U& rhs) const 
{
    Mat<T> lhs(*this);
    return lhs *= rhs;
}
template<typename T> template<typename U>                            
Mat<T> Mat<T>::operator/ (const U& rhs) const 
{
    Mat<T> lhs(*this);
    return lhs /= rhs;
}
t
#pragma region friend functions
template<typename T,typename U> 
Mat<T> operator+ (const U& lhs, const Mat<T>& mat)
{
    return mat + lhs;
}
template<typename T,typename U> 
Mat<T> operator- (const U& lhs, const Mat<T>& mat)
{
    return mat - lhs;
}
template<typename T,typename U> 
Mat<T> operator* (const U& lhs, const Mat<T>& mat)
{
    return mat * lhs;
}
template<typename T,typename U> 
Mat<T> operator/ (const U& lhs, const Mat<T>& mat)
{
    return mat / lhs;
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
	return det;
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
	return det;
}
template<typename T>
Mat<T> inv(const Mat<T>& mat)
{
	return inv_Gauss_Jordan(mat);
}
template<typename T>
Mat<T> inv_Gauss_Jordan(const Mat<T>& mat) {
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
#pragma endregion
#pragma endregion
#pragma endregion






#endif // MAT_H
