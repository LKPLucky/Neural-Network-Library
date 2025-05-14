#pragma once
#include <vector>
#include <string>
#include <filesystem>
#include <dlib\image_io.h>
#include <dlib\image_transforms.h>
#include <dlib\matrix.h>
#include <unordered_map>
#include <algorithm>
#include "..\headers\Tensor.hpp"

struct Dataset {
    vector<vector<Tensor>> data; // Each item: [image tensor, label tensor]
    vector<string> classLabels;  // Maps index -> class name
};

Tensor loadImageAsTensor(const string& imagePath);
Tensor one_hot_encode(int label, int numClasses);
Dataset loadDataFromFolder(const string& folderPath, int imageWidth, int imageHeight);
