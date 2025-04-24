#ifndef OUTPUTLAYER_H
#define OUTPUTLAYER_H

#include "Layer.h"

class OutputLayer : public Layer {
  std::function<VectorXd(const VectorXd&, const VectorXd&)> activation_function_deriv;

  public:
    OutputLayer(int input_size, int output_size, std::function<VectorXd(const VectorXd&)> activation_function, std::function<VectorXd(const VectorXd&, const VectorXd&)> activation_function_deriv);
    VectorXd get_deriv_activated_values(const VectorXd &target);
};

#endif