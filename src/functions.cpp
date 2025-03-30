#include "functions.h"

VectorXd sigmoid(const VectorXd &input) {
  return 1.0 / (1.0 + (-input.array()).exp());
}

VectorXd sigmoid_deriv(const VectorXd &input) {
  return sigmoid(input).array() * (1.0 - sigmoid(input).array());
}

double error(const VectorXd &output, const VectorXd &target) {
  return 0.5 * (output - target).squaredNorm();
}

VectorXd error_deriv(const VectorXd &output, const VectorXd &target) {
  return output - target;
}