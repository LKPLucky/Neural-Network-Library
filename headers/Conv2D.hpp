#pragma once
#include "..\headers\Layer.hpp"

class Conv2D : public Layer
{
private:
    Tensor InputCache;
    Tensor OutputCache_PreActivation;
    Tensor OutputCache_PostActivation;
    Tensor Weights;
    Tensor Bias;
    int Stride;
    bool FirstPass;
    string Activation;

    // INITIALIZE BIASES
    void InitializeBiases(double val = 0.01) override;
    // INITIALIZE WEIGHTS
    void InitializeWeights() override;

    // CONVOLUTION OPERATION
    void Convolution(Tensor& input);

    // Forward Pass
    void Forward(Tensor& input) override;

    // APPLY ACTIVATION FUNCTION
    void Activate(Tensor& input) override;

    // ADD BIASES
    void AddBias(Tensor& input) override;

public:

    // Constructor
    Conv2D (int F = 1, int KW = 1, int KH = 1, int KC = 1, int S = 1, string A = "none");

};