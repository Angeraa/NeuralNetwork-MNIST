#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <Eigen/dense>

using namespace Eigen;

VectorXd sigmoid(const VectorXd &input);

VectorXd sigmoid_deriv(const VectorXd &input);

VectorXd softmax(const VectorXd &input);

double error(const VectorXd &output, const VectorXd &target);

VectorXd softmax_error_deriv(const VectorXd &output, const VectorXd &target);

#endif