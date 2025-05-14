#pragma once
#include <vector>
#include <string>
#include <iostream>
using namespace std;

struct Tensor
{
    vector<double> Data;
    int Dim1; // Cols
    int Dim2; // Rows
    int Dim3; // Channels
    int Dim4; // Filter

    // Constructors
    Tensor() = default;
    Tensor(int D1, int D2 = 1, int D3 = 1, int D4 = 1);
    Tensor(const Tensor& other); // Copy Constructor (NOT USED)
  
    // Overload Operator () to access Data using indexes
    double& operator()(int x, int y = 0, int z = 0, int f = 0);

    void Display() const;
};

Tensor operator-(const Tensor& A, const Tensor& B);

Tensor operator*(Tensor& A, double val);