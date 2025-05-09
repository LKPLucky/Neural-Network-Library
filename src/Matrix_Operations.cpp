#include "..\headers\Matrix_Operations.hpp"

// Transpose a Matrix
vector<vector<double>> Transpose(vector<vector<double>> &matrix)
{
    // Get Rows and Cols of Matrix
    int rows = matrix.size();
    int cols = matrix[0].size();

    // Make a new matrix to store Transposed matrix temporarily
    vector<vector<double>> temp_matrix;

    // Transposing Matrix (Turning Rows into Cols) 
    vector<double> temp_row;
    for (int i = 0; i < cols; i++)
    {
        temp_row.clear();
        for (int j = 0; j < rows; j++)
        {
            temp_row.push_back(matrix[j][i]);
        }
        temp_matrix.push_back(temp_row);
    }

    return temp_matrix;
}

// Overload + Operator to add two vectors
vector<double> operator+(const vector<double>& vecA, const vector<double>& vecB)
{
    vector<double> temp_vec;
    if (vecA.size() > vecB.size())
    {
        for (int i = 0; i < vecB.size(); i++)
        {
            temp_vec.push_back(vecA[i] + vecB[i]);
        }
        for (int i = vecB.size(); i < vecA.size(); i++)
        {
            temp_vec.push_back(vecA[i]);
        }
    }
    else
    {
        for (int i = 0; i < vecA.size(); i++)
        {
            temp_vec.push_back(vecA[i] + vecB[i]);
        }
        for (int i = vecA.size(); i < vecB.size(); i++)
        {
            temp_vec.push_back(vecB[i]);
        }
    }

    return temp_vec;
}

// Dot Product
double DotProduct(vector<double> vecA, vector<double> vecB)
{
    double Result = 0.0;
    for (int i = 0; i < vecA.size(); i++)
    {
        Result += vecA[i] * vecB[i];
    }
    return Result;
}

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
        cout << "{ " << endl;
        for (int j = 0; j < input.Dim3; j++)
        {
            cout << "{ " << endl;
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
