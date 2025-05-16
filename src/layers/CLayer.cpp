#include "CLayer.h"
#include "functions.h"

#include <cmath>

CLayer::CLayer(int ksize, int depth, Dimension input_shape, std::function<MatrixXd(const MatrixXd&)> activation_function): depth(depth), input_shape(input_shape), activation_function(activation_function) {

  for (int i = 0; i < depth; i++) {
    std::vector<MatrixXd> kernel;
    for (int j = 0; j < input_shape.depth; j++) {
      MatrixXd kernel_slice = MatrixXd::Random(ksize, ksize);
      kernel.push_back(kernel_slice);
    }
    kernels.push_back(kernel);
  }
  output_shape = {input_shape.width - ksize + 1, input_shape.height- ksize + 1, depth};
  biases = std::vector<double>(depth, 0.0);
}

CLayer& CLayer::forward(const std::vector<MatrixXd> &inputs) {
  std::vector<MatrixXd> outputs;
  for (size_t i = 0; i < biases.size(); i++) {
    MatrixXd bias = MatrixXd::Constant(output_shape.height, output_shape.width, biases[i]);
    outputs.push_back(bias);
  }
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < input_shape.depth; j++) {
      outputs[i] += corr2D(inputs[j], kernels[i][j], output_shape);
    }
  }
  values = outputs;
  //activate the values and store them

  return *this;
}

std::vector<MatrixXd> CLayer::get_activated_values() {
  return activated_values;
}

CLayer& CLayer::set_delta(const MatrixXd &delta) {
  this->delta = delta;
  return *this;
}
MatrixXd CLayer::get_delta() {
  return delta;
}
std::vector<MatrixXd> CLayer::get_kernel(int index) {
  return kernels[index];
}
CLayer& CLayer::update_kernel(const MatrixXd &input, double learning_rate) {
  // Look up the math and put it here
  return *this;
}