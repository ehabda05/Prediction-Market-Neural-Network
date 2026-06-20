#include <iostream>
#include <vector>
#include "NeuralNetwork.hpp"

int main() {
    NeuralNetwork nn({2, 4, 1});

    std::vector<std::vector<double>> trainingData = {
        {0, 0},
        {0, 1},
        {1, 0},
        {1, 1}
    };

    std::vector<std::vector<double>> trainingLabels = {
        {0},
        {1},
        {1},
        {0}
    };

    nn.train(trainingData, trainingLabels, 10000, 0.5);

    std::cout << "Testing XOR neural network:\n\n";
    for (const auto& input : trainingData) {
        std::vector<double> output = nn.feedforward(input);

        int prediction = output[0] >= 0.5 ? 1 : 0;

        std::cout << input[0] << " XOR " << input[1]
                << " = " << prediction
                << " raw: " << output[0] << '\n';
    }
    return 0;
}