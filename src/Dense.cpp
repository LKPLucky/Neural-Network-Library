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

void Dense::Activate(Tensor& input)
{
    if (Activation == "relu")
    {
        OutputCache_PostActivation = ReLU(input);
    }
    else if (Activation == "softmax")
    {
        OutputCache_PostActivation = SoftMax(input);
    }
    else if (Activation == "sigmoid")
    {
        OutputCache_PostActivation = Sigmoid(input);
    }
    else // No Activvation Function used
    {
        OutputCache_PostActivation = input; 
    }
}

void Dense::AddBias(Tensor& input)
{
    for (int i = 0; i < input.Data.size(); i++)
    {
        input(i) = input(i) + Bias(i);
    }
}

void Dense::Forward(Tensor& input) 
{
    InputCache = input;
    if (FirstPass)
    {
        InitializeWeights();
        InitializeBiases();
    }
    OutputCache_PreActivation = input = MatMult(input, Weights);
    AddBias(input);
    Activate(input);
}
