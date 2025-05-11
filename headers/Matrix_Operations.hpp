#pragma once
#include <vector>
#include <iostream>
#include "..\headers\Tensor.hpp"
using namespace std;

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