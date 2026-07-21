#include<bits/stdc++.h>
#include "matrix.hpp"
#include "arena.hpp"

int main() {
    Arena* main_arena = arena_create(MiB(500));
    Matrix* test_images = MatrixLoad(main_arena, 60000, 784, "train_images.bin");

    arena_destroy(main_arena);
}