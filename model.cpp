#include "model.hpp"
#include "arena.hpp"

// TODO: implement layer_backward and model_backward functions
// TODO: implement claude fixes

Layer *model_add_layer(Model *model, LayerType type, int in_size,
                       int out_size) {
  assert(model->curr_layer < model->num_layers &&
         "model_add_layer: exceeded num_layers...");
  Layer *layer = &model->layers[model->curr_layer++];

  layer->type = type;
  layer->in_size = in_size;
  layer->out_size = out_size;

  return layer;
}

Layer *add_layer_dense(Arena *arena, Model *model, int in_size, int out_size) {
  Layer *layer = model_add_layer(model, LayerType::Dense, in_size, out_size);
  init_layer(layer, arena);

  return layer;
}

Layer *add_layer_relu(Arena *arena, Model *model, int size) {
  Layer *layer = model_add_layer(model, LayerType::ReLU, size, size);
  init_layer(layer, arena);

  return layer;
}

Layer *add_layer_softmax(Arena *arena, Model *model, int size) {
  Layer *layer = model_add_layer(model, LayerType::Softmax, size, size);
  init_layer(layer, arena);
  return layer;
}

void init_layer(Layer *layer, Arena *arena) {
  if (layer->type == LayerType::Dense) {
    // xavier init weights and 0 init biases soon
    layer->weights = matrix_create(arena, layer->in_size, layer->out_size);
    layer->biases = matrix_create(arena, 1, layer->out_size);

    float scale = sqrtf(2.0f / layer->in_size);

    for (int i = 0; i < layer->weights->size; i++) {
      layer->weights->data[i] = ((float)rand() / RAND_MAX * 2 - 1) * scale;
    }

    for (int i = 0; i < layer->biases->size; i++) {
      layer->biases->data[i] = 0.01f;
    }
  }

  else if (layer->type == LayerType::ReLU) {
    layer->weights = nullptr;
    layer->biases = nullptr;
  }

  else if (layer->type == LayerType::Softmax) {
    layer->weights = nullptr;
    layer->biases = nullptr;
  }

  else {
    cout << "Unknown layer type, terminating...";
    exit(1);
  }
}

Matrix *layer_forward(Arena *arena, Layer *layer, Matrix *input) {
  layer->input = input;

  Matrix *output = nullptr;
  if (layer->type == LayerType::Dense) {
    // y = x @ w + b
    Matrix *z = matrix_mult(arena, input, layer->weights);
    assert(z && "matrix_mult failed...");
    output = matrix_add_bias(arena, z, layer->biases);
    assert(output && "matrix_sum failed...");
    layer->output = output;
  }

  else if (layer->type == LayerType::ReLU) {
    output = matrix_relu(arena, input);
    layer->output = output;
  }

  else if (layer->type == LayerType::Softmax) {
    output = matrix_softmax(arena, input);
    layer->output = output;
  }

  else {
    cout << "Unknown layer type, terminating...";
    exit(1);
  }

  return output;
}

Matrix *layer_backward(Arena *arena, Layer *layer, Matrix *grad_output) {
  Matrix *grad_input = nullptr;

  if (layer->type == LayerType::Dense) {
    Matrix *input_t = matrix_transpose(arena, layer->input);
    layer->dweights = matrix_mult(arena, input_t, grad_output);
    assert(layer->dweights && "dweights: matrix_mult failed...");

    layer->dbiases = matrix_sum_rows(arena, grad_output);
    assert(layer->dbiases && "dbiases: matrix_sum_row failed...");

    Matrix *weights_t = matrix_transpose(arena, layer->weights);
    grad_input = matrix_mult(arena, grad_output, weights_t);
    assert(grad_input && "grad_input: matrix_mult failed...");
  }

  else if (layer->type == LayerType::ReLU) {
    grad_input = matrix_relu_grad(arena, layer->input, grad_output);
    assert(grad_input && "matrix_relu_grad failed...");
  }

  else if (layer->type == LayerType::Softmax) {
    // already accounted for in matrix_relu_grad
    grad_input = grad_output;
  }

  else {
    cout << "Unknown layer type, terminating...";
    exit(1);
  }

  return grad_input;
}

Model *model_create(Arena *arena, int num_layers) {
  Model *model = PUSH_STRUCT(arena, Model);
  model->num_layers = num_layers;
  model->layers = PUSH_ARRAY(arena, Layer, num_layers);

  return model;
}

Matrix *model_forward(Arena *arena, Model *model, Matrix *input) {
  Matrix *a = input;

  for (int i = 0; i < model->num_layers; i++) {
    a = layer_forward(arena, &model->layers[i], a);
  }

  return a;
}

Matrix *model_backward(Arena *arena, Model *model, Matrix *labels,
                       float learning_rate) {
  Layer *last_layer = &model->layers[model->num_layers - 1];

  // last layer is assumed to be softmax

  Matrix *grad = matrix_cross_entropy_grad(arena, last_layer->output, labels);
  assert(grad && "model_backward: initial grad failed");

  for (int i = model->num_layers - 1; i >= 0; i--) {
    Layer *layer = &model->layers[i];
    grad = layer_backward(arena, layer, grad);
    assert(grad && "model_backward: layer_backward failed");

    if (layer->type == LayerType::Dense) {
      float batch_size = (float)layer->input->rows;

      for (int k = 0; k < layer->dweights->size; k++) {
        layer->dweights->data[k] /= batch_size;
      }

      for (int k = 0; k < layer->dbiases->size; k++) {
        layer->dbiases->data[k] /= batch_size;
      }

      matrix_clip_inplace(layer->dweights, 1.0f);
      matrix_clip_inplace(layer->dbiases, 1.0f);

      for (int k = 0; k < layer->weights->size; k++) {
        layer->weights->data[k] -= learning_rate * layer->dweights->data[k];
      }

      for (int k = 0; k < layer->biases->size; k++) {
        layer->biases->data[k] -= learning_rate * layer->dbiases->data[k];
      }
    }
  }

  return grad;
}

float evaluate_accuracy(Arena *arena, Model *model, Matrix *inputs,
                        Matrix *labels) {
  arena_clear(arena);

  Matrix *predictions = model_forward(arena, model, inputs);

  int correct = 0;

  for (int i = 0; i < predictions->rows; i++) {
    int predicted = 0;
    float best = predictions->data[i * 10];

    for (int j = 1; j < 10; j++) {
      if (predictions->data[i * 10 + j] > best) {
        best = predictions->data[i * 10 + j];
        predicted = j;
      }
    }

    int actual = 0;

    for (int j = 0; j < 10; j++) {
      if (labels->data[i * 10 + j] == 1.0f) {
        actual = j;
        break;
      }
    }

    if (predicted == actual)
      correct++;
  }

  return 100.0f * (correct / (float)predictions->rows);
}
