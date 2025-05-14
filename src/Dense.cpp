#include "..\headers\Dense.hpp"

Dense::Dense(int out_c, string A) : OutChannels(out_c), Activation(A), Bias(Tensor(out_c)), FirstPass(true) {}

void Dense::InitializeBiases(double val)
{
    for (int i = 0; i < Bias.Data.size(); i++)
    {
        Bias(i) = val;
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
        ReLU(input);
    }
    else if (Activation == "sigmoid")
    {
        Sigmoid(input);
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
    input = MatMult(input, Weights);
    AddBias(input);
    Activate(input);
    OutputCache = input;
}

void Dense::ActivationDerivative(Tensor& d_out)
{
    if (Activation == "relu")
    {
        ReLUDerivative(d_out, OutputCache);
    }
    else if (Activation == "sigmoid")
    {
        SigmoidDerivative(d_out, OutputCache);
    }
}

void Dense::Backward(Tensor& d_out, const double LR)
{
    ActivationDerivative(d_out);
    UpdateWeights_and_Bias_and_DInput(d_out, LR);
}

void Dense::UpdateWeights_and_Bias_and_DInput(Tensor& d_out, const double LR)
{
    Tensor d_input(InputCache.Dim1);
    Tensor TempWeights(Weights);
    d_input.Data.assign(d_input.Data.size(), 0.0);
    for (int i = 0; i < Weights.Dim1; i++) // NO. OF NEURONS
    {
        for (int j = 0; j < Weights.Dim2; j++) // NO. OF WEIGHTS PER NEURON
        {
            TempWeights(i, j) -= (LR * (d_out(i) * InputCache(j)));
            d_input(j) += d_out(i) * Weights(i, j);
        }
        Bias(i) -= LR * d_out(i);
    }
    d_out = d_input;
    Weights = TempWeights;
}