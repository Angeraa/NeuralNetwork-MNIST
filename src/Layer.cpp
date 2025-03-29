#include "Layer.h"

Layer::Layer(int input_size, int output_size, std::function<VectorXd(const VectorXd&)> activation_function) : activation_function(activation_function) {
  weights = MatrixXd::Random(output_size, input_size);
  biases = VectorXd::Zero(output_size);
}

Layer& Layer::forward(const VectorXd &input) {
  VectorXd output = weights * input + biases;
  values = output;
  activated_values = activation_function(output);
  return *this;
}

VectorXd Layer::get_activated_values() {
  return activated_values;
}