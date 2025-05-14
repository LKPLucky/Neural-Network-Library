#include "..\headers\Tensor.hpp"

// Constructor
Tensor::Tensor(int D1, int D2, int D3, int D4) : Dim1(D1), Dim2(D2), Dim3(D3), Dim4(D4) 
{
    Data.resize(Dim1 * Dim2 * Dim3 * Dim4);
}

// Copy Constructor (NOT USED)
Tensor::Tensor(const Tensor& other) : Data(other.Data), Dim1(other.Dim1), Dim2(other.Dim2), Dim3(other.Dim3), Dim4(other.Dim4) {}


// Overload Operator () to access Data using indexes
double& Tensor::operator()(int x, int y, int z, int f)
{
    return Data[(f * Dim1 * Dim2 * Dim3) + (z * Dim1 * Dim2) + (y * Dim1) + x];
}

Tensor operator-(const Tensor& A, const Tensor& B)
{
    Tensor Result(A.Dim1, A.Dim2, A.Dim3, A.Dim4);

    for (int i = 0; i < A.Data.size(); ++i) {
        Result.Data[i] = A.Data[i] - B.Data[i];
    }

    return Result;
}

Tensor operator*(Tensor& A, double val)
{
    Tensor Result(A.Dim1, A.Dim2, A.Dim3, A.Dim4);
    for (int i = 0; i < A.Data.size(); ++i)
    {
        Result(i) = A(i) * val;  
    }
    return Result;
}


