#pragma once
#include "..\headers\Layer.hpp"
#include "..\headers\Image_Processing.hpp"
#include "..\headers\Loss_Functions.hpp"
#include <algorithm>
#include <random>
#include <string>

class Model {

private:
    vector<Layer*> LayerSequence;
    string LossFunction;
    string TrainingData;
    string ValidationData;
    vector<string> Labels;

    void Forward(Tensor& input);

    void Backward(Tensor& input, double LR);

    Tensor GetLogits(Tensor& input);

    Tensor FindLossDerivative(Tensor& O, Tensor& P);

    int getLabel(Tensor& output);

public:

    // Constructor
    Model(string LF, string TD, string VD);

    // Add Layer to Model
    template <typename T>
    void Add(T& obj)
    {
        LayerSequence.push_back(&obj);
    }

    void Train(int epochs, double LR);

    void Predict(string FP);
};