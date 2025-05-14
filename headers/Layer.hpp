#pragma once
#include <vector>
#include <string>

#include "..\headers\Tensor.hpp"
#include "..\headers\WeightsInit.hpp"
#include "..\headers\Matrix_Operations.hpp"
#include "..\headers\Activation_Functions.hpp"
#include <cctype>
#include <algorithm>

using namespace std;

class Layer {
    
public:
    virtual void Forward(Tensor& input) = 0;
    virtual void Backward(Tensor& d_out, const double LR) = 0;
    virtual void UpdateWeights_and_Bias_and_DInput(Tensor& d_out, const double LR) = 0;
    virtual void SetTraining(bool IsT) {};
};