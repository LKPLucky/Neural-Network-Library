#pragma once
#include "..\headers\Layer.hpp"

class MaxPool2D : public Layer
{
private:
    Tensor InputCache;
    int Stride;
    int KernelHeight;
    int KernelWidth;

    // FORWARD PASS
    void Forward(Tensor& input);

    void FindMax(Tensor& input);

    // INITIALIZE WEIGHTS
    virtual void InitializeWeights() override;
    // INITIALIZE BIASES
    virtual void InitializeBiases(double val) override;
    // APPLY ACTIVATION FUNCTION
    virtual void Activate(Tensor& input) override;
    // ADD BIASES
    virtual void AddBias(Tensor& input) override;

public:

    // CONSTRUCTOR
    MaxPool2D(int KH, int KW, int S);

};