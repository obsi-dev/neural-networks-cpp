#include "arena.hpp"
#include "matrix.hpp"
#include "model.hpp"
#include <bits/stdc++.h>

const string MODEL_PATH = "model.bin";

const int batch_size = 64;
const float learning_rate = 0.01f;

// Reads a line from stdin; empty line or EOF returns the default.
string prompt(const string &question, const string &default_answer) {
  cout << question << " [" << default_answer << "]: " << flush;
  string line;
  if (!getline(cin, line))
    return default_answer;

  size_t b = line.find_first_not_of(" \t\r\n");
  if (b == string::npos)
    return default_answer;
  size_t e = line.find_last_not_of(" \t\r\n");
  return line.substr(b, e - b + 1);
}

bool prompt_yes_no(const string &question, bool default_yes) {
  while (true) {
    string answer = prompt(question + (default_yes ? " (Y/n)" : " (y/N)"),
                           default_yes ? "y" : "n");
    char c = tolower(answer[0]);
    if (c == 'y')
      return true;
    if (c == 'n')
      return false;
    cout << "Please answer y or n." << endl;
  }
}

int prompt_int(const string &question, int default_value, int lo, int hi) {
  while (true) {
    string answer = prompt(question, to_string(default_value));
    try {
      size_t used;
      int v = stoi(answer, &used);
      if (used == answer.size() && v >= lo && v <= hi)
        return v;
    } catch (...) {
    }
    cout << "Please enter a number between " << lo << " and " << hi << "."
         << endl;
  }
}

// Loads images into data_arena and one-hot encodes the labels.
// The raw label buffer is a host temporary in scratch_arena.
void load_dataset(Arena *data_arena, Arena *scratch_arena, int n,
                  const string &images_path, const string &labels_path,
                  Matrix **images, Matrix **labels) {
  *images = matrix_load(data_arena, n, 784, images_path);
  *labels = matrix_create(data_arena, n, 10);

  arena_clear(scratch_arena);

  float *raw = PUSH_ARRAY(scratch_arena, float, n);
  float *one_hot = PUSH_ARRAY(scratch_arena, float, n * 10); // zeroed by arena

  ifstream file(labels_path, ios::binary);
  if (!file.is_open()) {
    cout << "Failed to open file: " << labels_path << " , exiting...";
    exit(1);
  }
  file.read((char *)raw, sizeof(float) * n);

  // perform one hot encoding on the labels
  for (int i = 0; i < n; i++) {
    int num = raw[i];
    one_hot[i * 10 + num] = 1.0f; // MNIST digits are zero-indexed by default
  }

  matrix_copy_from_host(*labels, one_hot);

  arena_clear(scratch_arena);
}

void train(Model *model, Arena *scratch_arena, Matrix *train_images,
           Matrix *train_labels, Matrix *batch_input, Matrix *batch_labels,
           int epochs) {
  const int num_batches = train_images->rows / batch_size;
  double total_seconds = 0.0;

  for (int e = 0; e < epochs; e++) {
    auto start = chrono::steady_clock::now();
    float epoch_loss = 0.0f;

    for (int b = 0; b < num_batches; b++) {
      arena_clear(scratch_arena);

      int start_row = b * batch_size;

      matrix_copy_rows(batch_input, train_images, start_row);
      matrix_copy_rows(batch_labels, train_labels, start_row);

      Matrix *predictions = model_forward(scratch_arena, model, batch_input);

      Matrix *losses =
          matrix_cross_entropy(scratch_arena, batch_labels, predictions);
      float batch_loss = matrix_sum_all(scratch_arena, losses);

      epoch_loss += batch_loss / batch_size;

      model_backward(scratch_arena, model, batch_labels, learning_rate);
    }

    double seconds =
        chrono::duration<double>(chrono::steady_clock::now() - start).count();

    total_seconds += seconds;

    cout << "Epoch " << e << " | Average Loss: " << (epoch_loss / num_batches)
         << " | Time: " << seconds << "s" << endl;
  }

  cout << "Training took " << total_seconds << "s total, "
       << total_seconds / epochs << "s per epoch on average" << endl;
}

int main() {
  srand((unsigned)time(nullptr));

  Arena *data_arena = arena_create(MiB(512));
  Arena *model_arena = arena_create(MiB(16));
  Arena *scratch_arena = arena_create(MiB(128));

  Model *model = model_create(model_arena, 4);
  add_layer_dense(model_arena, model, 784, 128);
  add_layer_relu(model_arena, model, 128);
  add_layer_dense(model_arena, model, 128, 10);
  add_layer_softmax(model_arena, model, 10);

  Matrix *batch_input = matrix_create(model_arena, batch_size, 784);
  Matrix *batch_labels = matrix_create(model_arena, batch_size, 10);

  // datasets are only loaded the first time they are needed
  Matrix *train_images = nullptr, *train_labels = nullptr;
  Matrix *test_images = nullptr, *test_labels = nullptr;

  auto train_and_maybe_save = [&]() {
    if (!train_images) {
      cout << "Loading training data..." << endl;
      load_dataset(data_arena, scratch_arena, 60000, "train_images.bin",
                   "train_labels.bin", &train_images, &train_labels);
    }

    int epochs = prompt_int("Number of epochs", 20, 1, 1000);

    model_reset_params(model_arena, model);
    train(model, scratch_arena, train_images, train_labels, batch_input,
          batch_labels, epochs);

    if (prompt_yes_no("Save weights to " + MODEL_PATH + "?", true)) {
      if (model_save(scratch_arena, model, MODEL_PATH))
        cout << "Saved." << endl;
    }
  };

  bool have_saved_model = ifstream(MODEL_PATH).good();

  if (have_saved_model &&
      prompt_yes_no("Found saved weights in " + MODEL_PATH + ". Load them?",
                    true)) {
    if (model_load(scratch_arena, model, MODEL_PATH)) {
      cout << "Loaded weights from " << MODEL_PATH << "." << endl;
    } else {
      cout << "Could not load weights, training from scratch." << endl;
      train_and_maybe_save();
    }
  } else {
    if (!have_saved_model)
      cout << "No saved weights found, training from scratch." << endl;
    train_and_maybe_save();
  }

  while (true) {
    cout << endl
         << "1) Test accuracy" << endl
         << "2) Retrain from scratch" << endl
         << "3) Save weights" << endl
         << "4) Quit" << endl;

    int choice = prompt_int("Choice", 4, 1, 4);

    if (choice == 1) {
      if (!test_images) {
        cout << "Loading test data..." << endl;
        load_dataset(data_arena, scratch_arena, 10000, "test_images.bin",
                     "test_labels.bin", &test_images, &test_labels);
      }

      float accuracy =
          evaluate_accuracy(scratch_arena, model, test_images, test_labels);
      cout << "Final model accuracy: " << accuracy << endl;
    } else if (choice == 2) {
      train_and_maybe_save();
    } else if (choice == 3) {
      if (model_save(scratch_arena, model, MODEL_PATH))
        cout << "Saved." << endl;
    } else {
      break;
    }
  }

  arena_destroy(data_arena);
  arena_destroy(model_arena);
  arena_destroy(scratch_arena);
}
