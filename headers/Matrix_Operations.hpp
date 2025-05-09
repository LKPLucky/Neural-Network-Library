#pragma once
#include <vector>
#include <iostream>
#include "..\headers\Tensor.hpp"
using namespace std;


// Transpose a Matrix
// vector<vector<double>> Transpose(vector<vector<double>> &matrix);

// Display a Matrix
void Display(vector<vector<double>>& matrix);

// Mat Mult
vector<vector<double>> MatMult(vector<vector<double>> &matrix1, vector<vector<double>> &matrix2);

// Overload + Operator to add two vectors
vector<double> operator+(const vector<double>& vecA, const vector<double>& vecB);

// Dot Product
double DotProduct(vector<double> vecA, vector<double> vecB);

Tensor MatMult(Tensor& matA, Tensor& matB);

void Display(Tensor& input);