#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include "..\headers\Tensor.hpp"
using namespace std;

// ReLU Activation Function
Tensor ReLU(Tensor& input);

// SoftMax Activation Function
Tensor SoftMax(Tensor& input);

// Sigmoid Activation Function
Tensor Sigmoid(Tensor& input);

void ReLUDerivative(Tensor& d_out, Tensor IO);

void SoftMaxDerivative(Tensor& d_out, Tensor IO);

void SigmoidDerivative(Tensor& d_out, Tensor IO);