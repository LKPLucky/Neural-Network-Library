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