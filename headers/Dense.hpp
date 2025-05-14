#pragma once
#include "..\headers\Layer.hpp"

class Dense : public Layer
{
private:
    Tensor InputCache;
    // Tensor OutputCache_PreActivation;
    Tensor OutputCache;
    Tensor Weights;
    Tensor Bias;
    string Activation;
    int OutChannels;
    bool FirstPass;

    // Forward Pass
    void Forward(Tensor& input) override;

    // BACK PROPOGATION
    void Backward(Tensor& d_out, const double LR) override;
    
    void UpdateWeights_and_Bias_and_DInput(Tensor& d_out, const double LR) override;

    // INITIALIZE BIASES
    void InitializeBiases(double val = 0.01);

    // INITIAILIZE WEIGHTS
    void InitializeWeights();

    // APPLY ACTIVATION FUNCTION
    void Activate(Tensor& input);

    // ADD BIASES
    void AddBias(Tensor& input);

    

    void ActivationDerivative(Tensor& d_out);

    

public:

    // Constructor
    Dense(int out_c, string A = "none");

};