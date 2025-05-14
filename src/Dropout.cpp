#include "..\headers\Dropout.hpp"

Dropout::Dropout(double DP) : DropoutProbability(DP), Scale(1 / (1 - DP)), IsTraining(false) { srand(time(0)); }

void Dropout::GenerateMask(Tensor& input)
{
    if (input.Dim3 > 1)
    {
        Mask.resize(input.Dim3);
        for (int i = 0; i < input.Dim3; i++)
        {
            Mask.push_back(rand() % 2);
        }
    }
    else
    {
        for (int i = 0; i < input.Dim1 * input.Dim2; i++)
        {
            Mask.push_back(rand() % 2);
        }
    }
}

void Dropout::Forward(Tensor&input)
{
    GenerateMask(input);
    if (input.Dim3 > 1)
    {
        for (int i = 0; i < Mask.size(); i++)
        {
            for (int j = 0; j < input.Dim1; j++)
            {
                for (int k = 0; k < input.Dim2; k++)
                {
                    input(k,j,i) *= Scale * Mask[i];
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < Mask.size(); i++)
        {
            input(i) *= Scale * Mask[i];
        }
    }
}

void Dropout::SetTraining(bool IsT) {IsTraining = IsT;}

void Dropout::Backward(Tensor& d_out, const double LR)
{
    UpdateWeights_and_Bias_and_DInput(d_out, LR);
}

void Dropout::UpdateWeights_and_Bias_and_DInput(Tensor& d_out, const double LR)
{
    if (d_out.Dim3 > 1)
    {
        for (int i = 0; i < Mask.size(); i++)
        {
            for (int j = 0; j < d_out.Dim1; j++)
            {
                for (int k = 0; k < d_out.Dim2; k++)
                {
                    d_out(k,j,i) *= Scale * Mask[i];
                }
            }
        }
    }
    else
    {
        for (int i = 0; i < Mask.size(); i++)
        {
            d_out(i) *= Scale * Mask[i];
        }
    }
}