#ifndef LAYER_H
#define LAYER_H

#include <Eigen/Dense>

using namespace Eigen;

class Layer {
  protected:
    MatrixXd weights;
    VectorXd biases;
    VectorXd values;
    VectorXd activated_values;
    std::function<VectorXd(const VectorXd&)> activation_function;

    VectorXd delta;

  public:
    Layer(int input_size, int output_size, std::function<VectorXd(const VectorXd&)> activation_function);
    virtual ~Layer() = default;

    Layer& forward(const VectorXd &input);

    VectorXd get_activated_values();

    virtual VectorXd get_deriv_activated_values(const VectorXd& = VectorXd()) = 0;

    Layer& set_delta(const VectorXd &delta);

    VectorXd get_delta();

    MatrixXd get_weights();

    Layer& update_weights(const VectorXd &input, double learning_rate);
};

#endif