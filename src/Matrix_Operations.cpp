#include "..\headers\Matrix_Operations.hpp"

// Mat Mult
Tensor MatMult(Tensor& matA, Tensor& matB)
{
    if (matA.Dim1 != matB.Dim2)
    {
        throw "Wrong Input to Dense Layer";
    }
    
    Tensor Result(matB.Dim1, matA.Dim2);
    double temp;
    for (int i = 0; i < matA.Dim2; i++)
    {
        for (int j = 0; j < matB.Dim1; j++)
        {
            temp = 0.0;
            for (int k = 0; k < matA.Dim1; k++)
            {
                temp +=  matA(k,i) * matB(j,k);
            }
            Result(j,i) = temp;
        }
    }
    return Result;
}

// Display a Tensor
void Display(Tensor& input)
{
    for (int i = 0; i < input.Dim4; i++)
    {
        cout << "{ ";
        for (int j = 0; j < input.Dim3; j++)
        {
            cout << "{ ";
            for (int k = 0; k < input.Dim2; k++)
            {
                cout << "{ ";
                for (int l = 0; l < input.Dim1; l++)
                {
                    cout << input(l, k, j, i) << " ";
                }
                cout << "}" << endl;
            }
            cout << "}" << endl;
        }
        cout << "}" << endl;
    }
}

// Find Horizontal Padding Amount
int FindPadH(const Tensor& input, const Tensor& Kernel, int Stride)
{
    int pad = Kernel.Dim1;
    while (pad < input.Dim1)
    {
        pad += Stride;
    }
    return pad % input.Dim1;
}

// Find Vertical Padding Amount
int FindPadV(const Tensor& input, const Tensor& Kernel, int Stride)
{
    int pad = Kernel.Dim2;
    while (pad < input.Dim2)
    {
        pad += Stride;
    }
    return pad % input.Dim2;
}

// Zero Padding
void ZeroPadding(Tensor& input, Tensor Kernel, int Stride)
{
    int pad = FindPadH(input, Kernel, Stride);
    if (pad != 0) // HORIZONTAL
    { 
        for (int i = 0; i < pad; i++)
        {
            for (int j = input.Data.size(); j > 0; j -= input.Dim1)
            {
                input.Data.insert(input.Data.begin() + j, 0.0);
            }
            input.Dim1++;
        }

    }
    pad = FindPadV(input, Kernel, Stride);
    if (pad != 0) // VERTICAL
    { 
        for (int i = 0; i < pad; i++)
        {
            for (int j = input.Data.size(); j > 0; j -= input.Dim1 * input.Dim2)
            {
                for (int k = 0; k < input.Dim1; k++)
                {
                    input.Data.insert(input.Data.begin() + j, 0.0);
                }
            }
            input.Dim2++;
        }

    }
}
