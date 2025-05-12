#pragma once
#include "..\headers\Layer.hpp"

class Dense : public Layer
{
private:
    Tensor InputCache;
    Tensor OutputCache_PreActivation;
    Tensor OutputCache_PostActivation;
    Tensor Weights;
    Tensor Bias;
    string Activation;
    int OutChannels;
    bool FirstPass;

    // Forward Pass
    void Forward(Tensor& input) override;

    // INITIALIZE BIASES
    void InitializeBiases(double val = 0.01) override;

    // INITIAILIZE WEIGHTS
    void InitializeWeights() override;

    // APPLY ACTIVATION FUNCTION
    void Activate(Tensor& input) override;

    // ADD BIASES
    void AddBias(Tensor& input) override;

public:

    // Constructor
    Dense(int out_c, string A = "none");

};