#include <iostream>
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

void NeuralNetwork::train(const std::vector<VectorXd> &inputs, std::vector<VectorXd> &targets, double learning_rate, int epochs) {
  std::cout << "===Begin Training===\n" << std::endl;
  for (int i = 0; i < epochs; i++) {
    double current_epoch_loss = 0.0;
    int correct = 0;
    for (size_t j = 0; j < inputs.size(); j++) {
      VectorXd input = inputs[j];
      VectorXd target = targets[j];

      run(input);

      VectorXd output = get_output();
      double loss = error(output, target);
      current_epoch_loss += loss;

      int predicted_index;
      int target_index;
      output.maxCoeff(&predicted_index);
      target.maxCoeff(&target_index);
      if (predicted_index == target_index) {
        correct++;
      }
      if (j < inputs.size() - 1 && j % 1000 == 0 && j != 0) {
        double ongoing_accuracy = static_cast<double>(correct) / j;
        std::cout << "Ongoing Training Accuracy: " << ongoing_accuracy << std::endl;
      }

      backward(input, target, learning_rate);
      if (j % 5000 == 0) {
        std::cout << "Epoch " << i + 1 << ": " << j << " out of " << inputs.size() << std::endl;
      }
    }
    current_epoch_loss /= inputs.size();
    std::cout << "Epoch " << i + 1 << " loss: " << current_epoch_loss << std::endl;
  }
}

void NeuralNetwork::test(const std::vector<VectorXd> &inputs, std::vector<VectorXd> &targets) {
  std::cout << "===Begin Testing===\n" << std::endl;
  int correct = 0;
  for (size_t i = 0; i < inputs.size(); i++) {
    VectorXd input = inputs[i];
    VectorXd target = targets[i];

    run(input);

    VectorXd output = get_output();

    int predicted_index;
    int target_index;
    output.maxCoeff(&predicted_index);
    target.maxCoeff(&target_index);
    if (predicted_index == target_index) {
      correct++;
    }
    if (i < inputs.size() - 1 && i % 200 == 0 && i != 0) {
      double ongoing_accuracy = static_cast<double>(correct) / i;
      std::cout << "Ongoing Testing Accuracy: " << ongoing_accuracy << std::endl;
    }
  }
  double accuracy = static_cast<double>(correct) / inputs.size();
  std::cout << "Final Accuracy: " << accuracy << std::endl;
}