#include "..\headers\Matrix_Operations.hpp"
#include "..\headers\Activation_Functions.hpp"
#include "..\headers\Tensor.hpp"
#include "..\headers\Dense.hpp"
#include "..\headers\Conv2D.hpp"
#include "..\headers\Model.hpp"

int main()
{
    Model A("CCE", "C:/Users/laksh/Desktop/IBA/Sem2/OOP/Project/TrainingData", "C:/Users/laksh/Desktop/IBA/Sem2/OOP/Project/ValidationData");
    A.Train(10, 0.01);
    A.Predict("C:/Users/laksh/Desktop/IBA/Sem2/OOP/Project/0_0_3521.jpeg");
}