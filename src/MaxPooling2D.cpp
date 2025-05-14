#include "..\headers\MaxPooling2D.hpp"

MaxPool2D::MaxPool2D(int KH, int KW, int S, string A) : KernelHeight(KH), KernelWidth(KW), Stride(S), Activation(A) {}

void MaxPool2D::Forward(Tensor& input)
{
    ZeroPadding(input, Tensor(KernelWidth, KernelHeight), Stride);
    InputCache = input;
    FindMax(input);
    Activate(input);
    OutputCache = input;
}

void MaxPool2D::FindMax(Tensor& input)
{
    input.Dim1 = ((input.Dim1 - KernelWidth) / Stride) + 1;
    input.Dim2 = ((input.Dim2 - KernelHeight) / Stride) + 1;
    input.Data.resize(input.Dim1 * input.Dim2 * input.Dim3 * input.Dim4, 0.0);
    input.Data.assign(input.Data.size(), 0);
    input.Data.shrink_to_fit();
    double max;
    for (int i = 0; i < input.Dim3; i++) // NO. OF CHANNELS PER FILTER
    {
        for (int j = 0; j < input.Dim2; j++) // HEIGHT OF OUTPUT
        {
            for (int k = 0; k < input.Dim1; k++) // WIDTH OF OUTPUT
            {
                int x_off = k * Stride;
                int y_off = j * Stride;
                max = -INFINITY;
                vector<int> temp = {};
                for (int m = 0; m < KernelHeight; m++)
                {
                    for (int n = 0; n < KernelWidth; n++)
                    {
                        if (max < InputCache(n + x_off, m + y_off, i))
                        {
                            max = InputCache(n + x_off, m + y_off, i);
                            temp.push_back(n + x_off);
                            temp.push_back(m + y_off);
                            temp.push_back(i);
                        }
                    }
                }
                Max_Coords.push_back(temp);
                input(k, j, i) = max;
            }
        }
    }
}

void MaxPool2D::Activate(Tensor& input)
{
    transform(Activation.begin(), Activation.end(), Activation.begin(),
    [](unsigned char c){ return tolower(c);});

    if (Activation == "relu")
    {
        OutputCache = ReLU(input);
    }
    else if (Activation == "sigmoid")
    {
        OutputCache = Sigmoid(input);
    }
    else if (Activation == "SoftMax")
    {
        OutputCache = SoftMax(input);
    }
    else // No Activvation Function used
    {
        OutputCache = input; 
    }
};

void MaxPool2D::ActivationDerivative(Tensor& d_out)
{
    if (Activation == "relu")
    {
        ReLUDerivative(d_out, OutputCache);
    }
    else if (Activation == "sigmoid")
    {
        SigmoidDerivative(d_out, OutputCache);
    }
    else if (Activation == "softmax")
    {
        SoftMaxDerivative(d_out, OutputCache);
    }
}

void MaxPool2D::Backward(Tensor& d_out, const double LR)
{
    ActivationDerivative(d_out);
    UpdateWeights_and_Bias_and_DInput(d_out, LR);
}

void MaxPool2D::UpdateWeights_and_Bias_and_DInput(Tensor& d_out, const double LR)
{
    Tensor d_input(InputCache);
    d_input.Data.assign(d_input.Data.size(), 0.0);
    for (int i = 0; i < Max_Coords.size(); i++)
    {
        d_input(Max_Coords[i][0], Max_Coords[i][1], Max_Coords[i][2]) = d_out(i);
    }
    d_out = d_input;
}