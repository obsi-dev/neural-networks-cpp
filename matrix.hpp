#pragma once

#include <bits/stdc++.h>
#include <math.h>
#include <iostream>
#include <stdlib.h>
#include <vector>

#include "arena.hpp"

typedef unsigned long long ull;

using namespace std;

class Matrix
{
public:
    int rows, cols;
    int size;
    float *data;

    Matrix(int r, int c) : rows(r), cols(c), size(r * c), data() {}

    float at(int r, int c) { return data[r * cols + c]; }
};

double power(double base, long long exp);

Matrix* MatrixCreate(Arena* arena, int rows, int cols);
Matrix* ReLU(Matrix* a);
Matrix* softmax(Matrix* a);
Matrix* MatrixSum(Matrix* a, Matrix* b);
Matrix* MatrixTranspose(Matrix* a);
Matrix* MatrixMult(Matrix* a, Matrix* b);
Matrix* MatrixLoad(Arena* arena, ull rows, ull cols, const string filepath);
