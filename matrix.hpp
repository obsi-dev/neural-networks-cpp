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

Matrix* matrix_create(Arena* arena, int rows, int cols);
Matrix* matrix_relu(Arena* arena, Matrix* a);
Matrix* matrix_softmax(Arena* arena, Matrix* a);
Matrix* matrix_sum(Arena* arena, Matrix* a, Matrix* b);
Matrix* matrix_transpose(Arena* arena, Matrix* a);
Matrix* matrix_mult(Arena* arena, Matrix* a, Matrix* b);
Matrix* matrix_load(Arena* arena, ull rows, ull cols, const string filepath);
Matrix* matrix_add_bias(Arena* arena, Matrix* input, Matrix* bias);
Matrix* matrix_cross_entropy(Arena* arena, const Matrix* p, const Matrix* q);
Matrix* matrix_relu_grad(Arena* arena, Matrix* layer_input, Matrix* grad_output);
Matrix* matrix_cross_entropy_grad(Arena* arena, Matrix* predictions, Matrix* labels);
