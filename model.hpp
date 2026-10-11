#pragma once

#include "arena.hpp"
#include "matrix.hpp"
#include <assert.h>
#include <bits/stdc++.h>
using namespace std;

enum class LayerType {
  Dense,
  ReLU,
  Softmax,
};

class Layer;

class Model {
public:
  int num_layers;
  int curr_layer;
  Layer *layers;

  Model(int num_layers) : num_layers(num_layers), curr_layer(0), layers() {}
};

class Layer {
public:
  LayerType type;
  int in_size;
  int out_size;

  Matrix *weights;
  Matrix *biases;

  Matrix *input;
  Matrix *output;

  Matrix *dweights;
  Matrix *dbiases;

  Layer(LayerType type, int in_size, int out_size)
      : type(type), in_size(in_size), out_size(out_size), weights(), biases(),
        input(), output(), dweights(), dbiases() {}
};

Matrix *layer_forward(Arena *arena, Layer *layer, Matrix *input);
Matrix *layer_backward(Arena *arena, Layer *layer, Matrix *grad_output);

Model *model_create(Arena *arena, int num_layers);

Matrix *model_forward(Arena *arena, Model *model, Matrix *input);
Matrix *model_backward(Arena *arena, Model *model, Matrix *labels,
                       float learning_rate);

void init_layer(Layer *layer, Arena *arena);

Layer *model_add_layer(Model *model, LayerType type, int in_size, int out_size);
Layer *add_layer_dense(Arena *arena, Model *model, int in_size, int out_size);
Layer *add_layer_relu(Arena *arena, Model *model, int size);
Layer *add_layer_softmax(Arena *arena, Model *model, int size);

float evaluate_accuracy(Arena *arena, Model *model, Matrix *inputs,
                        Matrix *labels);

// Re-randomise all dense layer params in place (for "retrain from scratch").
// arena is only used for temporary host staging.
void model_reset_params(Arena *arena, Model *model);

// Save / load weights and biases. scratch is used for temporary host buffers.
bool model_save(Arena *scratch, const Model *model, const string &path);
bool model_load(Arena *scratch, Model *model, const string &path);
