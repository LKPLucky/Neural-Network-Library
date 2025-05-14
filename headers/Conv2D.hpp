#pragma once
#include "..\headers\Layer.hpp"

class Conv2D : public Layer
{
private:
    Tensor InputCache;
    // Tensor OutputCache_PreActivation;
    Tensor OutputCache;
    Tensor Weights;
    Tensor Bias;
    int Stride;
    bool FirstPass;
    string Activation;

    // Forward Pass
    void Forward(Tensor& input) override;

    // Backward Pass
    void Backward(Tensor& d_out, const double LR) override;

    // Update Weights, Bias and get DInput
    void UpdateWeights_and_Bias_and_DInput(Tensor& d_out, const double LR) override;
    
    // INITIALIZE BIASES
    void InitializeBiases(double val = 0.01);
    // INITIALIZE WEIGHTS
    void InitializeWeights();

    // CONVOLUTION OPERATION
    void Convolution(Tensor& input);

    // APPLY ACTIVATION FUNCTION
    void Activate(Tensor& input);

    // ADD BIASES
    void AddBias(Tensor& input);

    void ActivationDerivative(Tensor& d_out);

public:

    // Constructor
    Conv2D (int F = 1, int KW = 1, int KH = 1, int KC = 1, int S = 1, string A = "none");

};