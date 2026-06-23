#include <bits/stdc++.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

//TODO: everything
//im killing myself

#define e 2.718

typedef unsigned long long ull;

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

Matrix* ReLU(Matrix* a) {
    int rows = a->rows, cols = a->cols;
    Matrix* result = new Matrix(rows, cols);
    for(int i = 0; i < rows * cols; i++) {
        result->data[i] = a->data[i] > 0 ? a->data[i] : 0.0;
    }

    return result;
}

double power(double base, long long exp) {
    if(exp == 0) return 1.0;

    bool isNegative = exp < 0;

    ull exponent = isNegative ? -exp : exp;

    double result = 1.0;

    while(exponent > 0) {
      if(exponent & 1) {
        result *= base;
      }

      base *= base;
      exponent >>= 1;
    }

    return isNegative ? 1 / result : result;
}

Matrix* softmax(Matrix* a) {
    int rows = a->rows, cols = a->cols;
    Matrix* result = new Matrix(rows, cols);
    
    double temp = 0;

    for(int i = 0; i < rows * cols; i++) {
        result->data[i] = power(e, a->data[i]);  
        temp += result->data[i];
    }
    
    for(int i = 0; i < rows * cols; i++) {
        result->data[i] /= temp;
    }

    return result;
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
