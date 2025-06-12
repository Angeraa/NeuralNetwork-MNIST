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

MatrixXd corr2D(const MatrixXd &input, const MatrixXd &kernel, const std::string &mode = "valid") {
  int in_h = input.rows(), in_w = input.cols();
  int k_h = kernel.rows(), k_w = kernel.cols();
  int out_h, out_w;
  MatrixXd padded_input = input;

  if (mode == "full") {
      int pad_h = k_h - 1, pad_w = k_w - 1;
      out_h = in_h + k_h - 1;
      out_w = in_w + k_w - 1;
      padded_input = MatrixXd::Zero(in_h + 2 * pad_h, in_w + 2 * pad_w);
      padded_input.block(pad_h, pad_w, in_h, in_w) = input;
  } else { // valid
      out_h = in_h - k_h + 1;
      out_w = in_w - k_w + 1;
  }

  MatrixXd output = MatrixXd::Zero(out_h, out_w);
  for (int i = 0; i < out_h; i++) {
      for (int j = 0; j < out_w; j++) {
          output(i, j) = (padded_input.block(i, j, k_h, k_w).array() * kernel.array()).sum();
      }
  }
  return output;
}