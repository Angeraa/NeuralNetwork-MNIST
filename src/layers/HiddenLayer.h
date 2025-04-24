#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

#include "Layer.h"

class HiddenLayer : public Layer {
  std::function<VectorXd(const VectorXd&)> activation_function_deriv;

  public:
    HiddenLayer(int input_size, int output_size, std::function<VectorXd(const VectorXd&)> activation_function, std::function<VectorXd(const VectorXd&)> activation_function_deriv);
    VectorXd get_deriv_activated_values(const VectorXd&) override;
};

#endif