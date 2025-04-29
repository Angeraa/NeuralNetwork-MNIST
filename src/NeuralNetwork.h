#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>
#include <memory>

#include "layers/Layer.h"

class NeuralNetwork {
  std::vector<std::unique_ptr<Layer>> layers;

  public:
    NeuralNetwork(std::vector<std::unique_ptr<Layer>> layers);
    NeuralNetwork& run(VectorXd& input);
    NeuralNetwork& backward(VectorXd &input, VectorXd &target, double learning_rate);
    VectorXd get_output();
    void train(const std::vector<VectorXd> &inputs, std::vector<VectorXd> &targets, double learning_rate, int epochs);
    void test(const std::vector<VectorXd> &inputs, std::vector<VectorXd> &targets);
};

#endif