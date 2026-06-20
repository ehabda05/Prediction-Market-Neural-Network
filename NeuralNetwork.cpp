#include "NeuralNetwork.hpp"

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes)
    : layerSizes(layerSizes)
{
    if (layerSizes.size() < 2) {
        throw std::invalid_argument("Neural network must have at least 2 layers.");
    }

    for (int size : layerSizes) {
        if (size <= 0) {
            throw std::invalid_argument("Layer sizes must be positive.");
        }
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (std::size_t layer = 1; layer < layerSizes.size(); ++layer) {
        int currentLayerSize = layerSizes[layer];
        int previousLayerSize = layerSizes[layer - 1];

        std::vector<std::vector<double>> layerWeights;
        std::vector<double> layerBiases;

        for (int neuron = 0; neuron < currentLayerSize; ++neuron) {
            std::vector<double> neuronWeights;

            for (int prevNeuron = 0; prevNeuron < previousLayerSize; ++prevNeuron) {
                neuronWeights.push_back(dist(gen));
            }

            layerWeights.push_back(neuronWeights);
            layerBiases.push_back(dist(gen));
        }

        weights.push_back(layerWeights);
        biases.push_back(layerBiases);
    }
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double NeuralNetwork::sigmoidDerivative(double x) {
    double s = sigmoid(x);
    return s * (1.0 - s);
}

std::vector<double> NeuralNetwork::feedforward(const std::vector<double>& input) {
    if (input.size() != static_cast<std::size_t>(layerSizes[0])) {
        throw std::invalid_argument("Input size does not match input layer size.");
    }

    std::vector<double> activations = input;

    for (std::size_t layer = 0; layer < weights.size(); ++layer) {
        std::vector<double> nextActivations;

        for (std::size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            double z = biases[layer][neuron];

            for (std::size_t prevNeuron = 0; prevNeuron < activations.size(); ++prevNeuron) {
                z += weights[layer][neuron][prevNeuron] * activations[prevNeuron];
            }

            nextActivations.push_back(sigmoid(z));
        }

        activations = nextActivations;
    }

    return activations;
}

void NeuralNetwork::train(const std::vector<std::vector<double>>& trainingData,
                          const std::vector<std::vector<double>>& trainingLabels,
                          int epochs,
                          double learningRate)
{
    if (trainingData.size() != trainingLabels.size()) {
        throw std::invalid_argument("Training data and labels must have the same size.");
    }

    if (epochs <= 0) {
        throw std::invalid_argument("Epochs must be positive.");
    }

    if (learningRate <= 0.0) {
        throw std::invalid_argument("Learning rate must be positive.");
    }

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (std::size_t i = 0; i < trainingData.size(); ++i) {
            backpropagate(trainingData[i], trainingLabels[i], learningRate);
        }
    }
}

void NeuralNetwork::backpropagate(const std::vector<double>& input,
                                  const std::vector<double>& expectedOutput,
                                  double learningRate)
{
    if (input.size() != static_cast<std::size_t>(layerSizes.front())) {
        throw std::invalid_argument("Input size does not match input layer size.");
    }

    if (expectedOutput.size() != static_cast<std::size_t>(layerSizes.back())) {
        throw std::invalid_argument("Expected output size does not match output layer size.");
    }

    std::vector<std::vector<double>> activations;
    std::vector<std::vector<double>> zValues;

    activations.push_back(input);

    for (std::size_t layer = 0; layer < weights.size(); ++layer) {
        std::vector<double> layerZ;
        std::vector<double> layerActivations;

        for (std::size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            double z = biases[layer][neuron];

            for (std::size_t prevNeuron = 0; prevNeuron < activations.back().size(); ++prevNeuron) {
                z += weights[layer][neuron][prevNeuron] * activations.back()[prevNeuron];
            }

            layerZ.push_back(z);
            layerActivations.push_back(sigmoid(z));
        }

        zValues.push_back(layerZ);
        activations.push_back(layerActivations);
    }

    std::vector<std::vector<double>> deltas(weights.size());

    int lastLayer = static_cast<int>(weights.size()) - 1;

    deltas[lastLayer].resize(layerSizes.back());

    for (std::size_t neuron = 0; neuron < activations.back().size(); ++neuron) {
        double output = activations.back()[neuron];
        double expected = expectedOutput[neuron];

        double error = output - expected;

        deltas[lastLayer][neuron] =
            error * sigmoidDerivative(zValues[lastLayer][neuron]);
    }

    for (int layer = lastLayer - 1; layer >= 0; --layer) {
        deltas[layer].resize(weights[layer].size());

        for (std::size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            double error = 0.0;

            for (std::size_t nextNeuron = 0; nextNeuron < weights[layer + 1].size(); ++nextNeuron) {
                error += weights[layer + 1][nextNeuron][neuron] *
                         deltas[layer + 1][nextNeuron];
            }

            deltas[layer][neuron] =
                error * sigmoidDerivative(zValues[layer][neuron]);
        }
    }

    for (std::size_t layer = 0; layer < weights.size(); ++layer) {
        for (std::size_t neuron = 0; neuron < weights[layer].size(); ++neuron) {
            biases[layer][neuron] -= learningRate * deltas[layer][neuron];

            for (std::size_t prevNeuron = 0; prevNeuron < weights[layer][neuron].size(); ++prevNeuron) {
                weights[layer][neuron][prevNeuron] -=
                    learningRate *
                    deltas[layer][neuron] *
                    activations[layer][prevNeuron];
            }
        }
    }
}