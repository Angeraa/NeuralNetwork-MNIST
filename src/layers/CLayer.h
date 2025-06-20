#ifndef CLAYER_H
#define CLAYER_H

#include <Eigen/Dense>
#include <vector>
#include "../utils/Dimension.h"

using namespace Eigen;

class CLayer {
  std::vector<std::vector<MatrixXd>> kernels;
  int ksize;
  std::vector<MatrixXd> biases;
  int depth;
  Dimension input_shape;
  Dimension output_shape;
  std::vector<MatrixXd> input;
  std::vector<MatrixXd> values;
  std::vector<MatrixXd> activated_values;
  std::function<MatrixXd(const MatrixXd&)> activation_function;

  MatrixXd delta;
  public:
    CLayer(int ksize, int depth, Dimension input_shape, std::function<MatrixXd(const MatrixXd&)> activation_function);

    CLayer& forward(const std::vector<MatrixXd> &inputs);
    std::vector<MatrixXd> get_activated_values();
    CLayer& set_delta(const MatrixXd &delta);
    MatrixXd get_delta();
    std::vector<MatrixXd> get_kernel(int index);
    CLayer& update_kernel(const std::vector<MatrixXd> &output__gradient, double learning_rate);
};

#endif