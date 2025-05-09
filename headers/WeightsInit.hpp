#pragma once
#include "..\headers\Tensor.hpp"
#include <cmath>
#include <random>

// GENERATE RANDOM NUMBERS FROM NORMAL AND UNIFORM DISTRIBUTION
double rand_normal(double mean, double stddev);
double rand_uniform(double min, double max);

// METHODS TO INITIALIZE WEIGHTS
void He_Init(Tensor& input, int fan_in); // Better for Layers with ReLU Activation
void Xavier_Init(Tensor& input, int fan_in, int fan_out); // Better for Layers with Sigmoid or Tanh Activation