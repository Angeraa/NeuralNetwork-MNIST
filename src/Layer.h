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
  std::function<VectorXd(const VectorXd&)> activation_function_deriv;

  VectorXd delta;

  public:
    Layer(int input_size, int output_size, std::function<VectorXd(const VectorXd&)> activation_function, std::function<VectorXd(const VectorXd&)> activation_function_deriv);

    Layer& forward(const VectorXd &input);

    VectorXd get_activated_values();

    VectorXd get_deriv_activated_values();

    Layer& set_delta(const VectorXd &delta);

    VectorXd get_delta();

    MatrixXd get_weights();

    Layer& update_weights(const VectorXd &input, double learning_rate);
};

#endif