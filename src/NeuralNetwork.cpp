#include "NeuralNetwork.h"
#include "functions.h"

NeuralNetwork::NeuralNetwork(std::vector<Layer> layers): layers(layers) {}

NeuralNetwork& NeuralNetwork::run(VectorXd &input) {
  layers[0].forward(input);
  for (size_t i = 1; i < layers.size(); i++) {
    layers[i].forward(layers[i - 1].get_activated_values());
  }
  return *this;
}

NeuralNetwork& NeuralNetwork::backward(VectorXd &input, VectorXd &target, double learning_rate) {
  VectorXd output_error = error_deriv(get_output(), target);
  VectorXd output_derivative = layers.back().get_deriv_activated_values();
  VectorXd output_delta = output_error.cwiseProduct(output_derivative);
  layers.back().set_delta(output_delta);
  for (int i = layers.size() - 2; i >= 0; i--) {
    MatrixXd next_weights = layers[i + 1].get_weights();
    VectorXd next_delta = layers[i + 1].get_delta();

    VectorXd error = next_weights * next_delta;
    VectorXd deriv = layers[i].get_deriv_activated_values();
    VectorXd delta = error.cwiseProduct(deriv);
    layers[i].set_delta(delta);
  }
  for (size_t i = 0; i < layers.size(); i++) {
    VectorXd input_;
    if (i == 0) input_ = input;
    else input_ = layers[i - 1].get_activated_values();
    layers[i].update_weights(input_, learning_rate);
  }
  return *this;
}

VectorXd NeuralNetwork::get_output() {
  return layers.back().get_activated_values();
}