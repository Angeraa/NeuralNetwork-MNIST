#ifndef READING_H
#define READING_H

#include <string>
#include <vector>
#include "Eigen/Dense"

void read_mnist_data(const std::string &path, std::vector<Eigen::VectorXd> &data);

void read_mnist_labels(const std::string &path, std::vector<Eigen::VectorXd> &labels);

#endif