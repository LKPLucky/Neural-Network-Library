#include "..\headers\Flatten.hpp"

Flatten::Flatten() {}

void Flatten::Forward(Tensor& input)
{
    InputShapeDim1 = input.Dim1;
    InputShapeDim2 = input.Dim2;
    InputShapeDim3 = input.Dim3;
    InputShapeDim4 = input.Dim4;
    input.Dim1 = InputShapeDim1 * InputShapeDim2 * InputShapeDim3 * InputShapeDim4;
    input.Dim2 = 1;
    input.Dim3 = 1;
    input.Dim4 = 1;
}

void Flatten::Backward(Tensor& d_out, const double LR)
{
    d_out.Dim1 = InputShapeDim1;
    d_out.Dim2 = InputShapeDim2;
    d_out.Dim3 = InputShapeDim3;
    d_out.Dim4 = InputShapeDim4;
}

void Flatten::UpdateWeights_and_Bias_and_DInput(Tensor& d_out, const double LR) {}