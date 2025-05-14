#include "..\headers\Model.hpp"

// Constructor
Model::Model(string LF, string TD, string VD) : LossFunction(LF), TrainingData(TD), ValidationData(VD) {}

void Model::Forward(Tensor& input)
{
    for (int i = 0; i < LayerSequence.size(); i++)
    {
        LayerSequence[i]->Forward(input);
    }
}

void Model::Backward(Tensor& d_out, double LR)
{
    for (int i = 0; i < LayerSequence.size(); i++)
    {
        LayerSequence[i]->Backward(d_out, LR);
    }
}

void Model::Train(int epochs, double LR)
{
    Dataset T = loadDataFromFolder(TrainingData, 32, 32);
    Dataset V = loadDataFromFolder(ValidationData, 32, 32);
    auto rng = default_random_engine {};
    double accuracy;
    for (int i = 0; i < epochs; i++)
    {
        
        shuffle(T.data.begin(), T.data.end(), rng);
        for (int j = 0; j < T.data.size(); j++)
        {
            Tensor Prediction = GetLogits(T.data[j][0]);
            Tensor d_out = FindLossDerivative(T.data[j][1], Prediction);
            Backward(d_out, LR);
        }
        shuffle(V.data.begin(), V.data.end(), rng);
        int correct = 0;
        for (int j = 0; j < V.data.size(); j++)
        {
            Tensor Prediction = GetLogits(V.data[j][0]);
            if (getLabel(Prediction) == getLabel(V.data[j][1]))
            {
                correct++;
            }
        }
        cout << "Epoch " << i << " Accuracy: " << static_cast<double>(correct / V.data.size()) << endl;
    }
}

Tensor Model::FindLossDerivative(Tensor& O, Tensor& P)
{
    Tensor d_out(P.Dim1);
    if (LossFunction == "BCE")
    {
        BinaryCrossEntropyLossDerivative(d_out, P, O);
    }
    else if (LossFunction == "CCE")
    {
        CategoricalCrossEntropyLossDerivative(d_out, P, O);
    }
    else if (LossFunction == "MSE")
    {
        MSELossDerivative(d_out, P, O);
    }
    return d_out;
}

Tensor Model::GetLogits(Tensor& input)
{
    Forward(input);
    return input;
}

int Model::getLabel(Tensor& output)
{
    int Label = 0;
    double maxVal = output(0);
    for (int i = 1; i < output.Dim1; ++i) 
    {
        if (output(i) > maxVal) 
        {
            maxVal = output(i);
            Label = i;
        }
    }
    return Label;
}

void Model::Predict(string FP)
{
    Tensor input = loadImageAsTensor(FP);
    Tensor output = GetLogits(input);
    cout << Labels[getLabel(output)] << endl;
}