#include "CLayer.h"
#include "functions.h"

#include <cmath>

CLayer::CLayer(int ksize, int depth, Dimension input_shape, std::function<MatrixXd(const MatrixXd&)> activation_function): ksize(ksize), depth(depth), input_shape(input_shape), activation_function(activation_function) {

  for (int i = 0; i < depth; i++) {
    std::vector<MatrixXd> kernel;
    for (int j = 0; j < input_shape.depth; j++) {
      MatrixXd kernel_slice = MatrixXd::Random(ksize, ksize);
      kernel.push_back(kernel_slice);
    }
    kernels.push_back(kernel);
  }
  output_shape = {input_shape.width - ksize + 1, input_shape.height- ksize + 1, depth};
  biases = std::vector<MatrixXd>(depth, MatrixXd::Zero(ksize, ksize));
}

CLayer& CLayer::forward(const std::vector<MatrixXd> &inputs) {
  // save input for backtrack
  this->input = inputs;
  std::vector<MatrixXd> outputs;
  for (size_t i = 0; i < biases.size(); i++) {
    outputs.push_back(biases[i]);
  }
  for (int i = 0; i < depth; i++) {
    for (int j = 0; j < input_shape.depth; j++) {
      outputs[i] += corr2D(inputs[j], kernels[i][j]);
    }
  }
  values = outputs;
  //activate the values and store them
  activated_values.clear();
  for (MatrixXd mat : outputs) {
    activated_values.push_back(activation_function(mat));
  }

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
CLayer& CLayer::update_kernel(const std::vector<MatrixXd> &output_gradient, double learning_rate) {
  std::vector<std::vector<MatrixXd>> kernel_gradient(depth, std::vector<MatrixXd>(input_shape.depth, MatrixXd::Zero(ksize, ksize)));
  std::vector<MatrixXd> input_gradient(input_shape.depth, MatrixXd::Zero(input_shape.height, input_shape.width));
  for (int i = 0; i < depth; i++) {
    for (int j = 0; i < input_shape.depth; j++) {
      kernel_gradient[i][j] = corr2D(input[j], output_gradient[i]);
      input_gradient[j] += corr2D(output_gradient[i], kernels[i][j], "full");
    }
  }
  
  for (int i = 0; i < depth; i++) {
    for (int j = 0; i < input_shape.depth; j++) {
      kernels[i][j] -= learning_rate * kernel_gradient[i][j];
    }
    biases[i] -= learning_rate * output_gradient[i];
  }

  return *this;
}