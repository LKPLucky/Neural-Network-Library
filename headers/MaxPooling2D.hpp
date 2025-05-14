#pragma once
#include "..\headers\Layer.hpp"

class MaxPool2D : public Layer
{
private:
    Tensor InputCache;
    int Stride;
    int KernelHeight;
    int KernelWidth;
    string Activation;
    Tensor OutputCache;
    vector<vector<int>> Max_Coords;

    // FORWARD PASS
    void Forward(Tensor& input) override;

    // BACKWARD PASS
    void Backward(Tensor& d_out, const double LR) override;

    // FIND OUTPUT
    void FindMax(Tensor& input);

    // APPLY ACTIVATION FUNCTION
    virtual void Activate(Tensor& input);

    // FIND ACTIVATION DERIVATIVE
    void ActivationDerivative(Tensor& d_out);

    void UpdateWeights_and_Bias_and_DInput(Tensor& d_out, const double LR) override;

public:

    // CONSTRUCTOR
    MaxPool2D(int KH, int KW, int S, string A = "none");

};