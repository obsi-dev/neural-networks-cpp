import gzip
import os
import struct
import urllib.request

import numpy as np

# Stable mirror of the original MNIST files (yann.lecun.com has become unreliable
# for scripted downloads, and some S3 mirrors 403 scripted/anonymous requests).
# This is a well-known GitHub mirror of the exact same dataset.
BASE_URL = "https://raw.githubusercontent.com/fgnt/mnist/master/"

FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}


def download(name, filename):
    if os.path.exists(filename):
        print(f"  {filename} already downloaded, skipping.")
        return
    url = BASE_URL + filename
    print(f"  downloading {url} ...")
    urllib.request.urlretrieve(url, filename)


def parse_idx_images(filename):
    # IDX image format: magic(4 bytes) num_images(4) rows(4) cols(4), then raw pixel bytes.
    with gzip.open(filename, "rb") as f:
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"unexpected magic number {magic} in {filename}"
        buf = f.read(num_images * rows * cols)
        data = np.frombuffer(buf, dtype=np.uint8)
        return data.reshape(num_images, rows * cols)


def parse_idx_labels(filename):
    # IDX label format: magic(4 bytes) num_labels(4), then raw label bytes.
    with gzip.open(filename, "rb") as f:
        magic, num_labels = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"unexpected magic number {magic} in {filename}"
        buf = f.read(num_labels)
        return np.frombuffer(buf, dtype=np.uint8)


def main():
    print("Downloading MNIST files...")
    for name, filename in FILES.items():
        download(name, filename)

    print("Parsing IDX files...")
    train_images = parse_idx_images(FILES["train_images"])
    train_labels = parse_idx_labels(FILES["train_labels"])
    test_images = parse_idx_images(FILES["test_images"])
    test_labels = parse_idx_labels(FILES["test_labels"])

    # Match the exact preprocessing/format the C++ side expects:
    # images -> float32, normalized to [0, 1], flattened rows
    # labels -> float32, raw digit value (0-9) -- main.cpp does the one-hot encoding itself
    train_images = train_images.astype(np.float32) / 255.0
    test_images = test_images.astype(np.float32) / 255.0
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)

    train_images.tofile("train_images.bin")
    train_labels.tofile("train_labels.bin")
    test_images.tofile("test_images.bin")
    test_labels.tofile("test_labels.bin")

    print(train_images.shape)
    print(train_labels.shape)
    print(test_images.shape)
    print(test_labels.shape)
    print("Done. Wrote train_images.bin, train_labels.bin, test_images.bin, test_labels.bin")


if __name__ == "__main__":
    main()
