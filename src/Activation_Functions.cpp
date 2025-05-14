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
    double max = *max_element(input.Data.begin(), input.Data.end());
    for(int i = 0; i < input.Data.size(); i++)
    {
        sum_of_exponentials += exp(input(i) - max);
    }
    for (int i = 0; i < input.Data.size(); i++)
    {
        input(i) = exp(input(i) - max)/sum_of_exponentials;
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

void SoftMaxDerivative(Tensor& d_out, Tensor IO)
{
    Tensor grad(d_out.Dim1, d_out.Dim2, d_out.Dim3, d_out.Dim4);

    for (int i = 0; i < d_out.Data.size(); i++)
    {
        double sum = 0.0;
        for (int j = 0; j < d_out.Data.size(); j++)
        {
            if (i == j)
                sum += d_out(j) * IO(i) * (1 - IO(j));
            else
                sum -= d_out(j) * IO(i) * IO(j);
        }
        grad(i) = sum;
    }
    d_out = grad;
}

void SigmoidDerivative(Tensor& d_out, Tensor IO)
{
    for (int i = 0; i < d_out.Data.size(); i++)
    {
        d_out(i) *= IO(i) * (1 - IO(i));
    }    
}