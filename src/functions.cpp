#include "functions.h"
#include <cmath>

VectorXd sigmoid(const VectorXd &input) {
  return 1.0 / (1.0 + (-input.array()).exp());
}

VectorXd sigmoid_deriv(const VectorXd &input) {
  return sigmoid(input).array() * (1.0 - sigmoid(input).array());
}

VectorXd softmax(const VectorXd &input) {
  VectorXd exp = (input.array() - input.maxCoeff()).exp();
  double sum_exp = exp.sum();
  return exp / sum_exp;
}

double error(const VectorXd &output, const VectorXd &target) {
  int target_index;
  target.maxCoeff(&target_index);
  return -log(output[target_index]);
}

VectorXd softmax_error_deriv(const VectorXd &output, const VectorXd &target) {
  return output - target;
}

MatrixXd corr2D(const MatrixXd &input, const MatrixXd &kernel, Dimension output_shape) {
  MatrixXd output = MatrixXd::Zero(output_shape.height, output_shape.width);
  for (int i = 0; i < output_shape.height; i++) {
    for (int j = 0; j < output_shape.width; j++) {
      output(i, j) = (input.block(i, j, kernel.rows(), kernel.cols()).array() * kernel.array()).sum();
    }
  }
  return output;
}