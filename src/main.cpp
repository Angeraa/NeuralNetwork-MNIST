#include <iostream>
#include <Eigen/Dense>
#include <memory>

#include "functions.h"
#include "NeuralNetwork.h"
#include "reading.h"
#include "layers/HiddenLayer.h"
#include "layers/OutputLayer.h"

using namespace std;
using namespace Eigen;

const string training_data_path = "data/MNIST/raw/train-images.idx3-ubyte";
const string training_labels_path = "data/MNIST/raw/train-labels.idx1-ubyte";
const string test_data_path = "data/MNIST/raw/t10k-images.idx3-ubyte";
const string test_labels_path = "data/MNIST/raw/t10k-labels.idx1-ubyte";

int main(void) {
  srand(time(0));

  vector<VectorXd> train_data;
  vector<VectorXd> train_labels;
  vector<VectorXd> test_data;
  vector<VectorXd> test_labels;

  read_mnist_data(training_data_path, train_data);
  read_mnist_labels(training_labels_path, train_labels);
  read_mnist_data(test_data_path, test_data);
  read_mnist_labels(test_labels_path, test_labels);

  vector<unique_ptr<Layer>> v;
  v.push_back(make_unique<HiddenLayer>(784, 64, sigmoid, sigmoid_deriv));
  v.push_back(make_unique<OutputLayer>(64, 10, softmax, softmax_error_deriv));

  NeuralNetwork network(move(v));

  network.train(train_data, train_labels, 0.1, 3);
  network.test(test_data, test_labels);

  return 0;
}