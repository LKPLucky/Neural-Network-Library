#pragma once
#include <vector>
#include <iostream>
#include "..\headers\Tensor.hpp"
using namespace std;


// Transpose a Matrix
// vector<vector<double>> Transpose(vector<vector<double>> &matrix);

// Overload + Operator to add two vectors
vector<double> operator+(const vector<double>& vecA, const vector<double>& vecB);

// Dot Product
double DotProduct(vector<double> vecA, vector<double> vecB);

// Mat Mult
Tensor MatMult(Tensor& matA, Tensor& matB);

// Display
void Display(Tensor& input);

// Find Horizontal Padding Amount
int FindPadH(const Tensor& input, const Tensor& Kernel, int Stride);

// Find Vertical Padding Amount
int FindPadV(const Tensor& input, const Tensor& Kernel, int Stride);

// Zero Padding
void ZeroPadding(Tensor& input, Tensor& Kernel, int Stride);