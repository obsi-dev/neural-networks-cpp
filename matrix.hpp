#include <bits/stdc++.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

using namespace std;

class Matrix {
  int rows, cols;
  vector<float> data;

  Matrix(int r, int c) : rows(r), cols(c), data(r * c, 0.0f) {}

  float at(int r, int c);
};

// float Matrix::at(int r, int c) {
//   return data[r * cols + c];
// }

Matrix MatrixMult(Matrix a, Matrix b, bool transpose_a, bool transpose_b);
Matrix relu(Matrix);
Matrix MatrixSum(Matrix a, Matrix b);
void MatTranpose(Matrix a);
