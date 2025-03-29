#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

using namespace Eigen;

class Layer {
  MatrixXd weights;
  VectorXd biases;
  VectorXd values;
  VectorXd activated_values;
  std::function<VectorXd(const VectorXd&)> activation_function;
  public:
    Layer(int input_size, int output_size, std::function<VectorXd(const VectorXd&)> activation_function);

    Layer& forward(const VectorXd &input);

    VectorXd get_activated_values();
};

#endif