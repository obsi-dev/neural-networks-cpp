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

echo "Compiling C++ files..."
g++ -std=c++20 -O2 -w arena.cpp \
    matrix.cpp \
    model.cpp \
    main.cpp \
    -o train
echo "Done."

echo ""

echo "Running training..."

./train
