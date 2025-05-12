#include "..\headers\MaxPooling2D.hpp"

MaxPool2D::MaxPool2D(int KH, int KW, int S) : KernelHeight(KH), KernelWidth(KW), Stride(S) {}

void MaxPool2D::Forward(Tensor& input)
{
    ZeroPadding(input, Tensor(KernelWidth, KernelHeight), Stride);
    InputCache = input;
    FindMax(input);
}

void MaxPool2D::FindMax(Tensor& input)
{
    input.Dim1 = ((input.Dim1 - KernelWidth) / Stride) + 1;
    input.Dim2 = ((input.Dim2 - KernelHeight) / Stride) + 1;
    input.Data.resize(input.Dim1 * input.Dim2 * input.Dim3 * input.Dim4, 0.0);
    input.Data.assign(input.Data.size(), 0);
    input.Data.shrink_to_fit();
    int x_off = 0;
    int y_off = 0;
    double max;
    for (int i = 0; i < input.Dim4; i++) // NO. OF FILTERS
    {
        for (int j = 0; j < input.Dim3; j++) // NO. OF CHANNELS PER FILTER
        {
            for (int k = 0; k < input.Dim2; k++) // HEIGHT OF OUTPUT
            {
                for (int l = 0; l < input.Dim1; l++) // WIDTH OF OUTPUT
                {
                    max = 0.0;
                    for (int m = 0; m < KernelHeight; m++)
                    {
                        for (int n = 0; n < KernelWidth; n++)
                        {
                            if (max < InputCache(n,m))
                            {
                                max = InputCache(n,m);
                            }
                        }
                    }
                    input(l,k) = max;
                    x_off += Stride;
                }
                y_off += Stride;
                x_off = 0;
            }
        }
    }
}

void MaxPool2D::InitializeWeights() {};
void MaxPool2D::InitializeBiases(double val) {};
void MaxPool2D::Activate(Tensor& input) {};
void MaxPool2D::AddBias(Tensor& input) {};


