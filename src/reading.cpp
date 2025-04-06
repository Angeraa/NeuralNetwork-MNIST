#include "reading.h"
#include <fstream>
#include <WinSock2.h>
#include "Eigen/Dense"


void read_mnist_data(const std::string &path, std::vector<Eigen::VectorXd> &data) {
  std::ifstream file(path, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int rows = 0;
    int cols = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ntohl(magic_number);
    file.read((char*)&number_of_images, sizeof(number_of_images));
    number_of_images = ntohl(number_of_images);
    file.read((char*)&rows, sizeof(rows));
    rows = ntohl(rows);
    file.read((char*)&cols, sizeof(cols));
    cols = ntohl(cols);

    for (int i = 0; i < number_of_images; i++) {
      Eigen::VectorXd datum(rows * cols);
      for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
          unsigned char temp = 0;
          file.read((char*)&temp, sizeof(temp));
          datum(r * cols + c) = (double)temp / 255.0;
        }
      }
      data.push_back(datum);
    }
  }
}

void read_mnist_labels(const std::string &path, std::vector<Eigen::VectorXd> &labels) {
  std::ifstream file(path, std::ios::binary);
  if (file.is_open()) {
    int magic_number = 0;
    int number_of_labels = 0;

    file.read((char*)&magic_number, sizeof(magic_number));
    magic_number = ntohl(magic_number);
    file.read((char*)&number_of_labels, sizeof(number_of_labels));
    number_of_labels = ntohl(number_of_labels);
    for (int i = 0; i < number_of_labels; i++) {
      unsigned char temp = 0;
      file.read((char*)&temp, sizeof(temp));
      Eigen::VectorXd label(10);
      label.setZero();
      label((int)temp) = 1.0;
      labels.push_back(label);
    }
  }
}