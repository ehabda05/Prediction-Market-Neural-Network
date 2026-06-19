#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <vector>
#include <cmath>
#include <stdexcept>

class NeuralNetwork {
  public:
    NeuralNetwork(const std::vector<int>& layerSizes);
    std::vector<double> feedforward(const std::vector<double>& input);
    void train(const std::vector<std::vector<double>>& trainingData,
               const std::vector<std::vector<double>>& trainingLabels,
               int epochs, double learningRate);

  private:
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> biases;

    std::vector<double> sigmoid(const std::vector<double>& z);
    std::vector<double> sigmoidDerivative(const std::vector<double>& z);
    void backpropagate(const std::vector<double>& input,
                       const std::vector<double>& expectedOutput,
                       double learningRate);
    };

#endif // NEURALNETWORK_HPP