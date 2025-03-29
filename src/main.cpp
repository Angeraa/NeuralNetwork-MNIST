#include <iostream>
#include <Eigen/Dense>

#include "Layer.h"

using namespace std;
using namespace Eigen;

VectorXd sigmoid(const VectorXd &v) {
  return 1.0 / (1.0 + (-v.array()).exp());
}

int main(void) {
  srand(time(0));

  
  return 0;
}