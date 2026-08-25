#include "arena.hpp"
#include "matrix.hpp"
#include "model.hpp"
#include <bits/stdc++.h>

int main() {
  srand((unsigned)time(nullptr));

  Arena *data_arena = arena_create(MiB(512));
  Arena *model_arena = arena_create(MiB(16));
  Arena *scratch_arena = arena_create(MiB(128));

  Matrix *train_images =
      matrix_load(data_arena, 60000, 784, "train_images.bin");
  Matrix *test_images = matrix_load(data_arena, 10000, 784, "test_images.bin");

  Matrix *train_labels = matrix_create(data_arena, 60000, 10);
  Matrix *test_labels = matrix_create(data_arena, 10000, 10);

  // perform one hot encoding on the training labels
  {
    Matrix *train_labels_file =
        matrix_load(data_arena, 60000, 1, "train_labels.bin");
    Matrix *test_labels_file =
        matrix_load(data_arena, 10000, 1, "test_labels.bin");

    for (int i = 0; i < 60000; i++) {
      int num = train_labels_file->data[i];
      train_labels->data[i * 10 + num] =
          1.0f; // MNIST digits are zero-indexed by default
    }

    for (int i = 0; i < 10000; i++) {
      int num = test_labels_file->data[i];
      test_labels->data[i * 10 + num] = 1.0f;
    }
  }

  Model *model = model_create(model_arena, 4);
  add_layer_dense(model_arena, model, 784, 128);
  add_layer_relu(model_arena, model, 128);
  add_layer_dense(model_arena, model, 128, 10);
  add_layer_softmax(model_arena, model, 10);

  const int batch_size = 64;
  const int num_batches = 60000 / batch_size;
  const int epochs = 20;
  const float learning_rate = 0.01f;

  Matrix *batch_input = matrix_create(model_arena, batch_size, 784);
  Matrix *batch_labels = matrix_create(model_arena, batch_size, 10);

  for (int e = 0; e < epochs; e++) {
    float epoch_loss = 0.0f;

    for (int b = 0; b < num_batches; b++) {
      arena_clear(scratch_arena);

      int start_row = b * batch_size;

      for (int r = 0; r < batch_size; r++) {
        memcpy(&batch_input->data[r * 784],
               &train_images->data[(start_row + r) * 784], sizeof(float) * 784);
        memcpy(&batch_labels->data[r * 10],
               &train_labels->data[(start_row + r) * 10], sizeof(float) * 10);
      }

      Matrix *predictions = model_forward(scratch_arena, model, batch_input);

      Matrix *losses =
          matrix_cross_entropy(scratch_arena, batch_labels, predictions);
      float batch_loss = 0.0f;

      for (int i = 0; i < losses->size; i++) {
        batch_loss += losses->data[i];
      }

      epoch_loss += batch_loss / batch_size;

      model_backward(scratch_arena, model, batch_labels, learning_rate);

      if (b == 0) {
        Layer *relu_layer = &model->layers[1];
        int dead_count = 0;
        for (int j = 0; j < relu_layer->output->cols; j++) {
          bool all_zero = true;
          for (int r = 0; r < relu_layer->output->rows; r++) {
            if (relu_layer->output->data[r * relu_layer->output->cols + j] >
                0) {
              all_zero = false;
              break;
            }
          }
          if (all_zero)
            dead_count++;
        }

        float w1_abs_sum = 0.0f;
        for (int k = 0; k < model->layers[0].weights->size; k++)
          w1_abs_sum += fabs(model->layers[0].weights->data[k]);

        float w2_abs_sum = 0.0f;
        for (int k = 0; k < model->layers[2].weights->size; k++)
          w2_abs_sum += fabs(model->layers[2].weights->data[k]);

        cout << "  epoch " << e << " diag: dead_relu=" << dead_count << "/"
             << relu_layer->output->cols << " |W1|_sum=" << w1_abs_sum
             << " |W2|_sum=" << w2_abs_sum << endl;
      }
    }

    cout << "Epoch " << e << "| average loss: " << (epoch_loss / num_batches)
         << endl;
  }

  float accuracy =
      evaluate_accuracy(scratch_arena, model, test_images, test_labels);

  cout << "Final model accuracy: " << accuracy << endl;

  arena_destroy(data_arena);
  arena_destroy(model_arena);
  arena_destroy(scratch_arena);
}
