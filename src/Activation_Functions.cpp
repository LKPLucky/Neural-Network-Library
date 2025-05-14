#include "..\headers\Activation_Functions.hpp"

Tensor ReLU(Tensor& input)
{
    for(int i = 0; i < input.Data.size(); i++)
    {
        if (input(i) < 0.0)
        {
            input(i) = 0.0;
        }
    }
    return input;
}

Tensor SoftMax(Tensor& input)
{
    double sum_of_exponentials = 0.0;
    for(int i = 0; i < input.Data.size(); i++)
    {
        sum_of_exponentials += exp(input(1));
    }
    for (int i = 0; i < input.Data.size(); i++)
    {
        input(i) = exp(input(i))/sum_of_exponentials;
    }
    return input;
}

Tensor Sigmoid(Tensor& input)
{
    for (int i = 0; i < input.Data.size(); i++)
    {
        input(i) = 1 / (1 + exp(-input(i)));
    }
    return input;
}

void ReLUDerivative(Tensor& d_out, Tensor IO)
{
    for (int i = 0; i < d_out.Data.size(); i++)
    {
        if (IO(i) == 0)
        {
            d_out(i) = 0.0;
        }
    }
}

void SoftMaxPlusCrossEntropyDerivative(Tensor& d_out, Tensor True)
{
    d_out = SoftMax(d_out);
    d_out = d_out - True;
}

void SigmoidDerivative(Tensor& d_out, Tensor IO)
{
    for (int i = 0; i < d_out.Data.size(); i++)
    {
        d_out(i) = d_out(i) * (1 - d_out(i));
    }    
}