#include <bits/stdc++.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

//TODO: everything
//im killing myself

using namespace std;

class Matrix {
public:
  int rows, cols;
  int size;
  vector<float> data;

  Matrix(int r, int c) : rows(r), cols(c), size(r * c), data(r * c, 0.0f) {}

  float at(int r, int c) { return data[r * cols + c]; }
};

Matrix MatrixMult(Matrix a, Matrix b, bool transpose_a, bool transpose_b);

Matrix relu(const Matrix &a) {
  Matrix result = a;
  for (int i = 0; i < a.rows * a.cols; i++) {
    result.data[i] = max(a.data[i], float(0));
  }
  return result;
}

void ReLU(Matrix a) {
    int rows = a.rows, cols = a.cols;
    for(int i = 0; i < rows * cols; i++) {
        if(a.data[i] < 0) a.data[i] = 0;
    }
}

bool MatrixSum(Matrix result, Matrix a, Matrix b) {
  if (a.rows != b.rows || a.cols != b.cols)
    return false;

  for (int i = 0; i < a.rows * a.cols; i++) {
    result.data[i] = a.data[i] + b.data[i];
  }
  return true;
}

Matrix MatTranpose(Matrix a) {
  Matrix result(a.cols, a.rows);
  for (int r = 0; r < a.rows; r++) {
    for (int c = 0; c < a.cols; c++) {
      result.data[r * result.cols + c] = a.at(c, r);
    }
  }
  return result;
}
