#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <random>

class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layerSizes);

    std::vector<double> feedforward(const std::vector<double>& input);

    void train(const std::vector<std::vector<double>>& trainingData,
               const std::vector<std::vector<double>>& trainingLabels,
               int epochs,
               double learningRate);

private:
    std::vector<int> layerSizes;

    // weights[layer][neuron][previousNeuron]
    std::vector<std::vector<std::vector<double>>> weights;

    // biases[layer][neuron]
    std::vector<std::vector<double>> biases;

    double sigmoid(double x);
    double sigmoidDerivative(double x);

    void backpropagate(const std::vector<double>& input,
                       const std::vector<double>& expectedOutput,
                       double learningRate);
};

#endif