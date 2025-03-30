#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include <vector>

#include "Layer.h"

class NeuralNetwork {
  std::vector<Layer> layers;

  public:
    NeuralNetwork(std::vector<Layer> layers);
    NeuralNetwork& run(VectorXd& input);
    VectorXd get_output();
};

#endif