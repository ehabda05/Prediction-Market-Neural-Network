#include "NeuralNetwork.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

void loadTrainingCSV(
    const std::string& filename,
    std::vector<std::vector<double>>& trainingData,
    std::vector<std::vector<double>>& trainingLabels
) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;

    // Skip header
    std::getline(file, line);

    int lineNumber = 1;

    while (std::getline(file, line)) {
        ++lineNumber;

        if (line.empty()) {
            continue;
        }

        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;

        while (std::getline(ss, cell, ',')) {
            if (!cell.empty()) {
                row.push_back(std::stod(cell));
            }
        }

        // 12 features + 1 label = 13 columns
        if (row.size() != 13) {
            std::cout << "Bad row on line " << lineNumber << ":\n";
            std::cout << line << "\n";
            std::cout << "Found " << row.size() << " columns, expected 13.\n";

            throw std::runtime_error("Invalid row size in CSV.");
        }

        std::vector<double> features(row.begin(), row.end() - 1);
        std::vector<double> label = { row.back() };

        trainingData.push_back(features);
        trainingLabels.push_back(label);
    }
}

void printLabelDistribution(
    const std::vector<std::vector<double>>& labels,
    const std::string& name
) {
    int yesCount = 0;
    int noCount = 0;

    for (const auto& label : labels) {
        if (label[0] >= 0.5) {
            yesCount++;
        } else {
            noCount++;
        }
    }

    std::cout << name << " YES labels: " << yesCount << "\n";
    std::cout << name << " NO labels: " << noCount << "\n";
}

void evaluateModel(
    NeuralNetwork& nn,
    const std::vector<std::vector<double>>& data,
    const std::vector<std::vector<double>>& labels,
    const std::string& name
) {
    int correct = 0;

    int trueYes = 0;
    int correctYes = 0;

    int trueNo = 0;
    int correctNo = 0;

    for (std::size_t i = 0; i < data.size(); ++i) {
        std::vector<double> prediction = nn.feedforward(data[i]);

        int predictedLabel = prediction[0] >= 0.5 ? 1 : 0;
        int actualLabel = labels[i][0] >= 0.5 ? 1 : 0;

        if (predictedLabel == actualLabel) {
            correct++;
        }

        if (actualLabel == 1) {
            trueYes++;

            if (predictedLabel == 1) {
                correctYes++;
            }
        } else {
            trueNo++;

            if (predictedLabel == 0) {
                correctNo++;
            }
        }

        std::cout << name
                  << " example " << i + 1
                  << " | probability YES: " << prediction[0]
                  << " | predicted: " << predictedLabel
                  << " | actual: " << actualLabel
                  << "\n";
    }

    double accuracy = static_cast<double>(correct) / data.size();

    std::cout << "\n" << name << " accuracy: "
              << accuracy * 100.0 << "%\n";

    std::cout << name << " YES accuracy: ";

    if (trueYes > 0) {
        std::cout << static_cast<double>(correctYes) / trueYes * 100.0 << "%\n";
    } else {
        std::cout << "No YES examples.\n";
    }

    std::cout << name << " NO accuracy: ";

    if (trueNo > 0) {
        std::cout << static_cast<double>(correctNo) / trueNo * 100.0 << "%\n";
    } else {
        std::cout << "No NO examples.\n";
    }
}

int main() {
    try {
        std::vector<std::vector<double>> allData;
        std::vector<std::vector<double>> allLabels;

        loadTrainingCSV("kalshi_training_data.csv", allData, allLabels);

        std::cout << "Loaded " << allData.size() << " examples.\n";

        if (allData.empty()) {
            throw std::runtime_error("No training examples loaded.");
        }

        printLabelDistribution(allLabels, "Full dataset");

        std::vector<std::pair<std::vector<double>, std::vector<double>>> examples;

        for (std::size_t i = 0; i < allData.size(); ++i) {
            examples.push_back({allData[i], allLabels[i]});
        }

        std::random_device rd;
        std::mt19937 gen(rd());

        std::shuffle(examples.begin(), examples.end(), gen);

        std::size_t splitIndex = static_cast<std::size_t>(examples.size() * 0.8);

        std::vector<std::vector<double>> trainData;
        std::vector<std::vector<double>> trainLabels;
        std::vector<std::vector<double>> testData;
        std::vector<std::vector<double>> testLabels;

        for (std::size_t i = 0; i < examples.size(); ++i) {
            if (i < splitIndex) {
                trainData.push_back(examples[i].first);
                trainLabels.push_back(examples[i].second);
            } else {
                testData.push_back(examples[i].first);
                testLabels.push_back(examples[i].second);
            }
        }

        std::cout << "\nTraining examples: " << trainData.size() << "\n";
        std::cout << "Testing examples: " << testData.size() << "\n\n";

        printLabelDistribution(trainLabels, "Train set");
        printLabelDistribution(testLabels, "Test set");

        std::cout << "\nTraining model...\n";

        // 12 input features, 1 output probability
        NeuralNetwork nn({12, 16, 8, 1});

        nn.train(trainData, trainLabels, 5000, 0.05);

        std::cout << "\nEvaluating on training data:\n";
        evaluateModel(nn, trainData, trainLabels, "Train");

        std::cout << "\nEvaluating on test data:\n";
        evaluateModel(nn, testData, testLabels, "Test");

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}