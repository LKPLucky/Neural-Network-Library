#pragma once
#include "..\headers\Layer.hpp"

class Dense : public Layer
{
//private:
public:
    Tensor InputCache;
    Tensor OutputCache;
    Tensor Weights;
    Tensor Bias;
    string Activation;
    int OutChannels;
    bool FirstPass;

//public:

    // Constructor
    Dense(int out_c, string A = "none");

    // Forward Pass
    void Forward(Tensor& input) override;

    // INITIALIZE BIASES
    void InitializeBiases(double val = 0.01) override;

    // INITIAILIZE WEIGHTS
    void InitializeWeights() override;

    // Calculate Output
};