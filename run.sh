#!/usr/bin/env bash
set -e # stop immediately if any step fails

echo "Preparing MNIST data..."

if [[ -f train_images.bin && -f train_labels.bin && -f test_images.bin && -f test_labels.bin ]]; then
    echo "Data files already exist, skipping regeneration."
    echo "(delete train_images.bin / train_labels.bin / test_images.bin / test_labels.bin to force a rebuild)"
else
    python3 data_handler.py
    echo "Done."
fi

echo ""

BACKEND="${1:-cpu}" # ./run.sh -> CPU build, ./run.sh cuda -> GPU build

echo "Compiling C++ files ($BACKEND)..."
if [[ "$BACKEND" == "cuda" ]]; then
    nvcc -std=c++17 -O2 -w -arch=native -DUSE_CUDA arena.cpp \
        matrix.cu \
        model.cpp \
        main.cpp \
        -o train
else
    g++ -std=c++20 -O2 -w arena.cpp \
        matrix.cpp \
        model.cpp \
        main.cpp \
        -o train
fi
echo "Done."

echo ""

echo "Running..."

./train
