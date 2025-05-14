#pragma once
#include "..\headers\Tensor.hpp"
#include <cmath>

double CategoricalCrossEntropyLoss(Tensor& predicted, Tensor& actual);
void CategoricalCrossEntropyLossDerivative(Tensor& d_out, Tensor& predicted, Tensor& actual);

double BinaryCrossEntropyLoss(Tensor& predicted, Tensor& actual);
void BinaryCrossEntropyLossDerivative(Tensor& d_out, Tensor& predicted, Tensor& actual);

double MSELoss(Tensor& predicted, Tensor& actual);
void MSELossDerivative(Tensor& d_out, Tensor& predicted, Tensor& actual);
