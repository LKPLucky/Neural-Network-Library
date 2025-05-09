#include "..\headers\Dense.hpp"

Dense::Dense(int out_c, string A) : OutChannels(out_c), Activation(A), Bias(Tensor(out_c)), FirstPass(true) {}

void Dense::InitializeBiases(double val)
{
    for (int i = 0; i < Bias.Data.size(); i++)
    {
        Bias.Data[i] = val;
    }
}

void Dense::InitializeWeights()
{
    Weights = Tensor(OutChannels, InputCache.Dim1); // Weights are stored per neuron coloumn wise.
    if (Activation == "relu") // USE He INITIALIZATION
    {
        He_Init(Weights, InputCache.Dim1);
    }
    else // USE XAVIER INITIAIZATION
    {
        Xavier_Init(Weights, InputCache.Dim1, OutChannels);
    }
    FirstPass = false;
}

void Dense::Forward(Tensor& input) 
{
    InputCache = input;
    if (FirstPass)
    {
        InitializeWeights();
        InitializeBiases();
    }
    OutputCache = input = MatMult(input, Weights);
}
