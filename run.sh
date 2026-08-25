#!/usr/bin/env bash
set -e  # stop immediately if any step fails

echo "=== Step 1: Preparing MNIST data (Python) ==="
# Only re-download/regenerate if the .bin files don't already exist,
# since data_handler.py downloads the full dataset via tensorflow_datasets.
if [[ -f train_images.bin && -f train_labels.bin && -f test_images.bin && -f test_labels.bin ]]; then
    echo "Data files already exist, skipping regeneration."
    echo "(delete train_images.bin / train_labels.bin / test_images.bin / test_labels.bin to force a rebuild)"
else
    python3 data_handler.py
fi

echo ""
echo "=== Step 2: Compiling C++ project ==="
g++ -std=c++20 -O2 -Wall \
    arena.cpp \
    matrix.cpp \
    model.cpp \
    main.cpp \
    -o train

echo ""
echo "=== Step 3: Running training ==="
./train
