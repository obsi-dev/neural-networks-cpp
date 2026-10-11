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

// Fills an already-allocated dense layer with random weights / constant biases.
// Values are generated in a temporary host buffer from the arena, copied in,
// and the buffer is popped again (host-only push, so pop_to is exact).
static void dense_fill_params(Arena *arena, Layer *layer) {
  ull saved_pos = arena->pos;

  float *w = PUSH_ARRAY(arena, float, layer->weights->size);
  float *b = PUSH_ARRAY(arena, float, layer->biases->size);

  float scale = sqrtf(2.0f / layer->in_size);

  for (int i = 0; i < layer->weights->size; i++) {
    w[i] = ((float)rand() / RAND_MAX * 2 - 1) * scale;
  }

  for (int i = 0; i < layer->biases->size; i++) {
    b[i] = 0.01f;
  }

  matrix_copy_from_host(layer->weights, w);
  matrix_copy_from_host(layer->biases, b);

  arena_pop_to(arena, saved_pos);
}

void init_layer(Layer *layer, Arena *arena) {
  if (layer->type == LayerType::Dense) {
    // xavier init weights and 0 init biases soon
    layer->weights = matrix_create(arena, layer->in_size, layer->out_size);
    layer->biases = matrix_create(arena, 1, layer->out_size);
    dense_fill_params(arena, layer);
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

      matrix_scale_inplace(layer->dweights, 1.0f / batch_size);
      matrix_scale_inplace(layer->dbiases, 1.0f / batch_size);

      matrix_clip_inplace(layer->dweights, 1.0f);
      matrix_clip_inplace(layer->dbiases, 1.0f);

      matrix_sgd_update(layer->weights, layer->dweights, learning_rate);
      matrix_sgd_update(layer->biases, layer->dbiases, learning_rate);
    }
  }

  return grad;
}

float evaluate_accuracy(Arena *arena, Model *model, Matrix *inputs,
                        Matrix *labels) {
  arena_clear(arena);

  Matrix *predictions = model_forward(arena, model, inputs);

  // host copies of predictions and labels (temporary, freed with the arena)
  float *pred = PUSH_ARRAY(arena, float, predictions->size);
  float *lbl = PUSH_ARRAY(arena, float, labels->size);
  matrix_copy_to_host(predictions, pred);
  matrix_copy_to_host(labels, lbl);

  int correct = 0;

  for (int i = 0; i < predictions->rows; i++) {
    int predicted = 0;
    float best = pred[i * 10];

    for (int j = 1; j < 10; j++) {
      if (pred[i * 10 + j] > best) {
        best = pred[i * 10 + j];
        predicted = j;
      }
    }

    int actual = 0;

    for (int j = 0; j < 10; j++) {
      if (lbl[i * 10 + j] == 1.0f) {
        actual = j;
        break;
      }
    }

    if (predicted == actual)
      correct++;
  }

  return 100.0f * (correct / (float)predictions->rows);
}

void model_reset_params(Arena *arena, Model *model) {
  for (int i = 0; i < model->num_layers; i++) {
    if (model->layers[i].type == LayerType::Dense)
      dense_fill_params(arena, &model->layers[i]);
  }
}

// ---------------- save / load ----------------
// The file is just the raw float32 parameters of every Dense layer, in layer
// order: W1 | b1 | W2 | b2 | ... The architecture comes from the code, so the
// only check needed on load is that the file has exactly the right size.

bool model_save(Arena *scratch, const Model *model, const string &path) {
  ofstream out(path, ios::binary);
  if (!out.is_open()) {
    cout << "model_save: failed to open " << path << endl;
    return false;
  }

  ull saved_pos = scratch->pos;

  for (int i = 0; i < model->num_layers; i++) {
    Layer *layer = &model->layers[i];
    if (layer->type != LayerType::Dense)
      continue;

    float *w = PUSH_ARRAY(scratch, float, layer->weights->size);
    float *b = PUSH_ARRAY(scratch, float, layer->biases->size);
    matrix_copy_to_host(layer->weights, w);
    matrix_copy_to_host(layer->biases, b);
    out.write((const char *)w, sizeof(float) * layer->weights->size);
    out.write((const char *)b, sizeof(float) * layer->biases->size);
  }

  arena_pop_to(scratch, saved_pos);

  out.close();
  if (out.fail()) {
    cout << "model_save: write failed" << endl;
    return false;
  }

  return true;
}

bool model_load(Arena *scratch, Model *model, const string &path) {
  ifstream in(path, ios::binary | ios::ate); // open positioned at the end
  if (!in.is_open()) {
    return false;
  }

  ull expected_size = 0;
  for (int i = 0; i < model->num_layers; i++) {
    Layer *layer = &model->layers[i];
    if (layer->type == LayerType::Dense)
      expected_size +=
          sizeof(float) * (layer->weights->size + layer->biases->size);
  }

  ull file_size = (ull)in.tellg();
  if (file_size != expected_size) {
    cout << "model_load: expected " << expected_size << " bytes, file has "
         << file_size << endl;
    return false;
  }

  in.seekg(0, ios::beg);

  ull saved_pos = scratch->pos;

  for (int i = 0; i < model->num_layers; i++) {
    Layer *layer = &model->layers[i];
    if (layer->type != LayerType::Dense)
      continue;

    float *w = PUSH_ARRAY(scratch, float, layer->weights->size);
    float *b = PUSH_ARRAY(scratch, float, layer->biases->size);
    in.read((char *)w, sizeof(float) * layer->weights->size);
    in.read((char *)b, sizeof(float) * layer->biases->size);
    matrix_copy_from_host(layer->weights, w);
    matrix_copy_from_host(layer->biases, b);
  }

  arena_pop_to(scratch, saved_pos);
  return true;
}
