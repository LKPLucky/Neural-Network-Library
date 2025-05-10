#include "..\headers\WeightsInit.hpp"

// Generate Random Number from Normal Distribution
double rand_normal(double mean, double stddev)
{
    static random_device rd;
    static std::mt19937 gen(rd());
    normal_distribution<> dist(mean, stddev);  // Normal distribution

    return dist(gen);
}

// Generate Random Number from Uniform Distribution
double rand_uniform(double min, double max) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_real_distribution<> dist(min, max);
    return dist(gen);
}

void He_Init(Tensor& input, int fan_in)
{
    double stddev = sqrt(2.0 / fan_in);
    for (int i = 0; i < input.Data.size(); i++)
    {
        input(i) = rand_normal(0, stddev);
    }
}


void Xavier_Init(Tensor& input, int fan_in, int fan_out)
{
    double limit = sqrt(6.0 / (fan_in + fan_out));
    for (int i = 0; i < input.Data.size(); i++)
    {
        input(i) = rand_uniform(-limit, limit);
    }
}
