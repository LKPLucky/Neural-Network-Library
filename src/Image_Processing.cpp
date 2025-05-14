#include "..\headers\Image_Processing.hpp"

Tensor loadImageAsTensor(const string& imagePath) {
    dlib::matrix<dlib::rgb_pixel> img_rgb;
    dlib::matrix<unsigned char> img_gray;

    try {
        dlib::load_image(img_rgb, imagePath); 
    } catch (...) {
        dlib::load_image(img_gray, imagePath); 
    }

    Tensor tensor(32, 32, 3);  

    if (!img_rgb.size() == 0) {
        // Resize RGB image
        dlib::matrix<dlib::rgb_pixel> resized;
        dlib::resize_image(32.0 / img_rgb.nr(), img_rgb, resized);

        for (int y = 0; y < 32; ++y) {
            for (int x = 0; x < 32; ++x) {
                const dlib::rgb_pixel& pixel = resized(y, x);
                tensor(x, y, 0) = pixel.red / 255.0;
                tensor(x, y, 1) = pixel.green / 255.0;
                tensor(x, y, 2) = pixel.blue / 255.0;
            }
        }
    } else {
        // Grayscale
        dlib::matrix<unsigned char> resized;
        dlib::resize_image(32.0 / img_gray.nr(), img_gray, resized);
        tensor = Tensor(32, 32, 1, 1);  // 1 channel

        for (int y = 0; y < 32; ++y) {
            for (int x = 0; x < 32; ++x) {
                tensor(x, y, 0, 0) = resized(y, x) / 255.0;
            }
        }
    }

    return tensor;
}

Tensor one_hot_encode(int label, int numClasses) {
    Tensor result(numClasses, 1, 1, 1);
    for (int i = 0; i < numClasses; ++i)
        result(i, 0, 0, 0) = (i == label) ? 1.0 : 0.0;
    return result;
}

Dataset loadDataFromFolder(const std::string& folderPath, int imageWidth, int imageHeight) {
    Dataset dataset;
    std::unordered_map<std::string, int> labelToIndex;
    int currentLabelIndex = 0;

    for (const auto& entry : std::filesystem::directory_iterator(folderPath)) {
        if (entry.is_directory()) {
            std::string labelName = entry.path().filename().string();
            if (labelToIndex.find(labelName) == labelToIndex.end()) {
                labelToIndex[labelName] = currentLabelIndex++;
                dataset.classLabels.push_back(labelName);
            }

            int labelIndex = labelToIndex[labelName];

            for (const auto& file : std::filesystem::directory_iterator(entry.path())) {
                Tensor inputTensor = loadImageAsTensor(file.path().string());
                Tensor labelTensor = one_hot_encode(labelIndex, labelToIndex.size());
                dataset.data.push_back({inputTensor, labelTensor});
            }
        }
    }

    return dataset;
}