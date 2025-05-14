#pragma once
#include "..\headers\Layer.hpp"

class Flatten : public Layer
{
private:
    int InputShapeDim1;
    int InputShapeDim2;
    int InputShapeDim3;
    int InputShapeDim4;
    

    void Forward(Tensor& input) override;

    void Backward(Tensor& d_out, const double LR) override;

    void UpdateWeights_and_Bias_and_DInput(Tensor& input, const double LR) override;

public:

    Flatten();

};