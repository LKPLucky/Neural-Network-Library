#pragma once
#include <vector>
#include <string>
#include "..\headers\Tensor.hpp"
#include "..\headers\WeightsInit.hpp"
#include "..\headers\Matrix_Operations.hpp"

using namespace std;

class Layer {
    
public:

    // INITIALIZE WEIGHTS
    virtual void InitializeWeights() = 0;
    // INITIALIZE BIASES
    virtual void InitializeBiases(double val) = 0;


    virtual void Forward(Tensor& input) = 0;
    // virtual void back_propogation();
    
};