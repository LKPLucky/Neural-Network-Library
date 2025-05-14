#include "..\headers\Loss_Functions.hpp"

double CategoricalCrossEntropyLoss(Tensor& predicted, Tensor& actual)
{
    double loss = 0.0;
    for (int i = 0; i < predicted.Data.size(); i++)
    {
        double y = actual(i);
        if (y == 1.0)
        {
            double y_hat = min(max(predicted(i), 1e-7), 1.0);  // Clamp
            loss += -log(y_hat);
        }
    }
    return loss;
}
void CategoricalCrossEntropyLossDerivative(Tensor& d_out, Tensor& predicted, Tensor& actual)
{
    for (int i = 0; i < predicted.Data.size(); i++)
    {
        d_out(i) = - actual(i) / max(predicted(i), 1e-7);
    }
}


double BinaryCrossEntropyLoss(Tensor& predicted, Tensor& actual)
{
    double loss = 0.0;
    for (int i = 0; i < predicted.Data.size(); i++)
    {
        double y = actual(i);
        double y_hat = predicted(i);
        y_hat = min(max(y_hat, 1e-7), 1 - 1e-7);
        loss += - (y * log(y_hat) + (1 - y) * log(1 - y_hat));
    }
    return loss / predicted.Data.size();
}
void BinaryCrossEntropyLossDerivative(Tensor& d_out, Tensor& predicted, Tensor& actual)
{
    for (int i = 0; i < predicted.Data.size(); i++)
    {
        double y = actual(i);
        double y_hat = min(max(predicted(i), 1e-7), 1 - 1e-7);  // Clamp
        d_out(i) = (y_hat - y) / (y_hat * (1 - y_hat)) / predicted.Data.size();
    }
}


double MSELoss(Tensor& predicted, Tensor& actual)
{
    double loss = 0.0;
    for (int i = 0; i < predicted.Data.size(); i++)
    {
        double diff = predicted(i) - actual(i);
        loss += diff * diff;
    }
    return loss / predicted.Data.size();
}
void MSELossDerivative(Tensor& d_out, Tensor& predicted, Tensor& actual)
{
    for (int i = 0; i < predicted.Data.size(); i++)
    {
        d_out(i) = 2.0 * (predicted(i) - actual(i)) / predicted.Data.size();
    }
}
