#include "Layer.h"

Layer::Layer(int input_size,
    int output_size,
    std::function<VectorXd(const VectorXd&)> activation_function) :
      activation_function(activation_function) {
      weights = MatrixXd::Random(input_size, output_size);
      biases = VectorXd::Zero(output_size);
}

Layer& Layer::forward(const VectorXd &input) {
  VectorXd output = weights.transpose() * input + biases;
  values = output;
  activated_values = activation_function(output);
  return *this;
}

VectorXd Layer::get_activated_values() {
  return activated_values;
}

Layer& Layer::set_delta(const VectorXd &delta) {
  this->delta = delta;
  return *this;
}

VectorXd Layer::get_delta() {
  return delta;
}

MatrixXd Layer::get_weights() {
  return weights;
}

Layer& Layer::update_weights(const VectorXd &input, double learning_rate) {
  weights = weights - (learning_rate * input * delta.transpose());
  biases = biases - (learning_rate * delta);
  return *this;
}