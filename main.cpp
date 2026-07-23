#include<bits/stdc++.h>
#include "matrix.hpp"
#include "arena.hpp"

int main() {
    Arena* main_arena = arena_create(MiB(500));
    
    Matrix* train_images = MatrixLoad(main_arena, 60000, 784, "train_images.bin");
    Matrix* test_images = MatrixLoad(main_arena, 10000, 784, "test_images.bin");

    Matrix* train_labels = MatrixCreate(main_arena, 60000, 10);
    Matrix* test_labels = MatrixCreate(main_arena, 10000, 10);
    
    //perform one hot encoding on the training labels
    {
        Matrix* train_labels_file = MatrixLoad(main_arena, 60000, 1, "train_labels.bin");
        Matrix* test_labels_file = MatrixLoad(main_arena, 10000, 1, "test_labels.bin");

        for(int i = 0; i < 60000; i++) {
            int num = train_labels_file->data[i];
            train_labels->data[i * 10 + num] = 1.0f; //MNIST digits are zero-indexed by default
        }

        for(int i = 0; i < 10000; i++) {
            int num = test_labels_file->data[i];
            test_labels->data[i * 10 + num] = 1.0f;
        }
    }
    
    arena_destroy(main_arena);
}