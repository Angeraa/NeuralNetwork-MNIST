#include "OutputLayer.h"

OutputLayer::OutputLayer(int input_size, 
  int output_size, 
  std::function<VectorXd(const VectorXd&)> activation_function, 
  std::function<VectorXd(const VectorXd&, const VectorXd&)> activation_function_deriv) :
    Layer(input_size, output_size, activation_function),
    activation_function_deriv(activation_function_deriv) {}

VectorXd OutputLayer::get_deriv_activated_values(const VectorXd &target) {
  return activation_function_deriv(activation_function(values), target); 
}