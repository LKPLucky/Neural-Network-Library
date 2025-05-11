#include "..\headers\Conv2D.hpp"

Conv2D::Conv2D(int F, int KW, int KH, int KC, int S, string A) : Weights(Tensor(KW, KH, KC, F)), Stride(S), FirstPass(true), Activation(A) {}

void Conv2D::Forward(Tensor& input)
{
    ZeroPadding(input, Weights, Stride);
    InputCache = input;

    // INITIALIZE WEIGHTS IF FIRST PASS
    if (FirstPass)
    {
        InitializeWeights();
        InitializeBiases();
    }
    Convolution(input);
    AddBias(input);
    OutputCache_PreActivation = input;
    Activate(input);
}

void Conv2D::InitializeBiases(double val)
{
    Bias.Dim1 = ((InputCache.Dim1 - Weights.Dim1) / Stride) + 1;
    Bias.Dim2 = ((InputCache.Dim2 - Weights.Dim2) / Stride) + 1;
    Bias.Dim3 = InputCache.Dim4;
    Bias.Dim4 = 1;
    Bias.Data.resize(Bias.Dim1 * Bias.Dim2 * Bias.Dim3, val);
}

void Conv2D::InitializeWeights()
{
    if (Activation == "relu") // USE He INITIALIZATION
    {
        He_Init(Weights, Weights.Dim1 * Weights.Dim2 * Weights.Dim3);
    }
    else // USE XAVIER INITIAIZATION
    {
        Xavier_Init(Weights, Weights.Dim1 * Weights.Dim2 * Weights.Dim3, Weights.Dim1 * Weights.Dim2 * Weights.Dim4);
    }
    FirstPass = false;
}

void Conv2D::Convolution(Tensor& input)
{
    input.Dim1 = ((input.Dim1 - Weights.Dim1) / Stride) + 1;
    input.Dim2 = ((input.Dim2 - Weights.Dim3) / Stride) + 1;
    input.Dim3 = Weights.Dim4;
    input.Dim4 = 1;
    input.Data.resize(input.Dim1 * input.Dim2 * input.Dim3 * input.Dim4, 0.0);
    input.Data.assign(input.Data.size(), 0);
    input.Data.shrink_to_fit();
    int x_off = 0;
    int y_off = 0;
    for (int i = 0; i < Weights.Dim4; i++) // NO. OF FILTERS
    {
        for (int j = 0; j < InputCache.Dim3; j++) // NO. OF MATRICES PER FILTER
        {
            for (int k = 0; k < input.Dim2; k++) // OUTPUT ROWS
            {
                for (int l = 0; l < input.Dim1; l++) // OUTPUT COLS
                {
                    for (int m = 0; m < Weights.Dim2; m++) // KERNEL ROWS
                    {
                        for (int n = 0; n < Weights.Dim1; n++) // KERNEL COLS
                        {
                            input(l,k,i) += InputCache(n + x_off, m + y_off, j) * Weights(n, m, j, i);  
                        }
                    }
                    x_off += Stride;
                }
                y_off += Stride;
                x_off = 0;
            }
            y_off = 0;
        }
    }    
}

void Conv2D::Activate(Tensor& input)
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

void Conv2D::AddBias(Tensor& input)
{   
    for (int i = 0; i < input.Data.size(); i++)
    {
        input(i) = input(i) + Bias(i);
    }
}


