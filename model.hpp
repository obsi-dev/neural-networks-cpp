#pragma once

#include<bits/stdc++.h>
#include "matrix.hpp"
#include "arena.hpp"
using namespace std;

enum class LayerType {
    Dense,
    ReLU,
    Softmax,
};

class Model {
    public:
    int num_layers;
    Layer* layers;

    Model(int num_layers) : num_layers(num_layers), layers() {}
};

class Layer {
    public:
    LayerType type;
    int in_size;
    int out_size;

    Matrix* weights;
    Matrix* biases;

    Matrix* input;
    Matrix* output;

    Matrix* dweights;
    Matrix* dbiases;

    Layer(LayerType type, int in_size, int out_size) : type(type), in_size(in_size), out_size(out_size), weights(), biases(), input(), output(), dweights(), dbiases() {}
};

Layer* make_layer_dense(Arena* arena, int in_size, int out_size);
Layer* make_layer_relu(Arena* arena, int size);

Matrix* layer_forward(Layer* layer, Matrix* input);
Matrix* layer_backward(Layer* layer, Matrix* grad_output);

Model* model_create(Arena* arena, int num_layers);
Matrix* model_forward(Model* model, Matrix* input);
Matrix* model_backward(Model* model, Matrix* grad_loss, float learning_rate);