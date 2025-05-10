#pragma once
#include <vector>
#include <cmath>
#include "..\headers\Tensor.hpp"
using namespace std;

// ReLU Activation Function
Tensor ReLU(Tensor& input);

// SoftMax Activation Function
Tensor SoftMax(Tensor& input);

// Sigmoid Activation Function
Tensor Sigmoid(Tensor& input);