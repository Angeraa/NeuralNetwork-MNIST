#include "NeuralNetwork.h"

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers): layers(layers) {}

NeuralNetwork& NeuralNetwork::run(VectorXd &input) {
  layers[0].forward(input);
  for (size_t i = 1; i < layers.size(); i++) {
    layers[i].forward(layers[i - 1].get_activated_values());
  }
  return *this;
}

VectorXd NeuralNetwork::get_output() {
  return layers.back().get_activated_values();
}