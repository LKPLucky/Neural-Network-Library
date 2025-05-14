#include "..\headers\Conv2D.hpp"

Conv2D::Conv2D(int F, int KW, int KH, int KC, int S, string A) : Weights(Tensor(KW, KH, KC, F)), Bias(F), Stride(S), FirstPass(true), Activation(A) {}

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
    Activate(input);
}

void Conv2D::InitializeBiases(double val)
{
    Bias.Data.assign(Bias.Data.size(), val);
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
    input.Dim2 = ((input.Dim2 - Weights.Dim2) / Stride) + 1;
    input.Dim3 = Weights.Dim4;
    input.Dim4 = 1;
    input.Data.resize(input.Dim1 * input.Dim2 * input.Dim3 * input.Dim4, 0.0);
    input.Data.assign(input.Data.size(), 0);
    input.Data.shrink_to_fit();
    int x_off = 0;
    int y_off = 0;
    for (int i = 0; i < Weights.Dim4; i++) // NO. OF FILTERS
    {
        for (int j = 0; j < InputCache.Dim3; j++) // NO. OF CHANNELS PER FILTER
        {
            for (int k = 0; k < input.Dim2; k++) // OUTPUT ROWS
            {
                int x_off = k * Stride;
                int y_off = j * Stride;
                for (int l = 0; l < input.Dim1; l++) // OUTPUT COLS
                {
                    for (int m = 0; m < Weights.Dim2; m++) // KERNEL ROWS
                    {
                        for (int n = 0; n < Weights.Dim1; n++) // KERNEL COLS
                        {
                            input(l,k,i) += InputCache(n + x_off, m + y_off, j) * Weights(n, m, j, i);  
                        }
                    }
                }
            }
        }
    }    
}

void Conv2D::Activate(Tensor& input)
{
    if (Activation == "relu")
    {
        OutputCache = ReLU(input);
    }
    else if (Activation == "sigmoid")
    {
        OutputCache = Sigmoid(input);
    }
    else // No Activvation Function used
    {
        OutputCache = input; 
    }
}

void Conv2D::AddBias(Tensor& input)
{   
    for (int i = 0; i < input.Dim3; i++)
    {
        for (int j = 0; j < input.Dim2; j++)
        {
            for (int k = 0; k < input.Dim1; i++)
            {
                input(k, j, i) += Bias(i);
            }
        }
    }
}

void Conv2D::UpdateWeights_and_Bias_and_DInput(Tensor& d_out, const double LR)
{
    Tensor d_input(InputCache.Dim1, InputCache.Dim2, InputCache.Dim3);
    d_input.Data.assign(d_input.Data.size(), 0.0);
    Tensor TempWeights(Weights.Dim1, Weights.Dim2, Weights.Dim3, Weights.Dim4);
    TempWeights.Data.assign(TempWeights.Data.size(), 0.0);
    for (int i = 0; i < Weights.Dim4; i++) // NO. OF FILTERS
    {
        double sum = 0;
        for (int j = 0; j < d_out.Dim2; j++) // NO. OF ROWS IN OUTPUT OF CONV LAYER
        {
            for (int k = 0; k < d_out.Dim1; k++) // NO. OF COLS IN OUTPUT OF CONV LAYER
            {
                sum += d_out(k, j, i);
                int x_off = k * Stride;
                int y_off = j * Stride;
                for (int l = 0; l < Weights.Dim3; l++) // NO. OF CHANNELS PER FILTERS
                {
                    for (int m = 0; m < Weights.Dim2; m++) // KERNEL ROWS
                    {
                        for (int n = 0; n < Weights.Dim1; n++) // KERNEL COLS
                        {
                            TempWeights(n, m, l, i) += InputCache(n + x_off, m + y_off, l) * d_out(k,j,i);
                            d_input(n + x_off, m + y_off, l) += Weights(n, m, l, i) * d_out(k, j, i);
 
                        }
                    }
                }
            }
        }
        Bias(i) -= LR * sum;
    }
    Weights = Weights - (TempWeights * LR);
    d_out = d_input;
}

void Conv2D::ActivationDerivative(Tensor& d_out)
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

void Conv2D::Backward(Tensor& d_out, const double LR)
{
    ActivationDerivative(d_out);
    UpdateWeights_and_Bias_and_DInput(d_out, LR);
}