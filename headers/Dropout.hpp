#pragma once
#include "..\headers\Layer.hpp"

class Dropout : public Layer
{
private:
    double DropoutProbability;
    double Scale;
    bool IsTraining;
    vector<double> Mask;

    void GenerateMask(Tensor& input);
    void Forward(Tensor& input) override;
    void Backward(Tensor& d_out, const double LR) override;
    void UpdateWeights_and_Bias_and_DInput(Tensor& d_out, const double LR) override;
    void SetTraining(bool IsT) override;

public:

    Dropout(double DP = 0.5);
};