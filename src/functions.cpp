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