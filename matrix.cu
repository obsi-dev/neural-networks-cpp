
// CUDA backend: implements exactly the same API as matrix.cpp (matrix.hpp).
// Build with -DUSE_CUDA and compile this file *instead of* matrix.cpp.
// Every Matrix has a host-side header (rows/cols/size readable on the CPU)
// and a ->data pointer into GPU memory.
#include "matrix.hpp"
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err_ = (call);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),    \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Kernel launches don't return errors, so check right after each launch.
// Build with -DCUDA_SYNC_DEBUG to also catch errors raised *inside* a kernel
// at the exact launch that caused them (slow, debugging only).
#ifdef CUDA_SYNC_DEBUG
#define KERNEL_CHECK() CUDA_CHECK(cudaDeviceSynchronize())
#else
#define KERNEL_CHECK() CUDA_CHECK(cudaGetLastError())
#endif

static const int THREADS = 256;
static inline int blocks_for(int n) { return (n + THREADS - 1) / THREADS; }

// ---------------------------------------------------------------- kernels

__global__ void k_relu(const float *a, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] > 0.0f ? a[i] : 0.0f;
}

__global__ void k_add(const float *a, const float *b, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] + b[i];
}

__global__ void k_add_bias(const float *in, const float *bias, float *out,
                           int n, int cols) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = in[i] + bias[i % cols];
}

__global__ void k_relu_grad(const float *in, const float *grad, float *out,
                            int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = in[i] > 0.0f ? grad[i] : 0.0f;
}

__global__ void k_sub(const float *a, const float *b, float *out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    out[i] = a[i] - b[i];
}

__global__ void k_cross_entropy(const float *p, const float *q, float *out,
                                int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float epsilon = 1e-7f;
    float q_clipped = fmaxf(epsilon, fminf(q[i], 1.0f - epsilon));
    out[i] = p[i] == 0.0f ? 0.0f : -p[i] * logf(q_clipped);
  }
}

__global__ void k_clip(float *a, float max_abs, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] = fminf(max_abs, fmaxf(-max_abs, a[i]));
}

__global__ void k_scale(float *a, float s, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    a[i] *= s;
}

__global__ void k_sgd(float *w, const float *dw, float lr, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    w[i] -= lr * dw[i];
}

// One thread per row. Fine for 10 classes; for wide rows you would use one
// warp per row and do the max / sum with warp shuffles.
__global__ void k_softmax(const float *a, float *out, int rows, int cols) {
  int r = blockIdx.x * blockDim.x + threadIdx.x;
  if (r >= rows)
    return;

  const float *in = a + (size_t)r * cols;
  float *o = out + (size_t)r * cols;

  float row_max = in[0];
  for (int c = 1; c < cols; c++)
    row_max = fmaxf(row_max, in[c]);

  float row_sum = 0.0f;
  for (int c = 0; c < cols; c++) {
    o[c] = expf(in[c] - row_max);
    row_sum += o[c];
  }

  for (int c = 0; c < cols; c++)
    o[c] /= row_sum;
}

// Column sums (dbiases). One thread per column; neighbouring threads read
// neighbouring addresses in each row, so the loads are coalesced.
__global__ void k_sum_rows(const float *a, float *out, int rows, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols)
    return;
  float sum = 0.0f;
  for (int r = 0; r < rows; r++)
    sum += a[(size_t)r * cols + c];
  out[c] = sum;
}

__global__ void k_transpose(const float *a, float *out, int rows, int cols) {
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows && c < cols)
    out[(size_t)c * rows + r] = a[(size_t)r * cols + c];
}

// Naive matmul: one thread per output element. Every thread re-reads a full
// row of A and a full column of B from global memory. Build with
// -DNAIVE_MATMUL to use this one (useful as a baseline).
__global__ void k_matmul_naive(const float *A, const float *B, float *C, int M,
                               int K, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N) {
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
      acc += A[row * K + k] * B[k * N + col];
    C[row * N + col] = acc;
  }
}

// Tiled matmul: each block computes a TILE x TILE patch of C. It walks along
// K one tile at a time: the block cooperatively loads a tile of A and a tile
// of B into shared memory (one element per thread), syncs, and then every
// thread does TILE multiply-adds out of fast shared memory. Global memory
// reads drop by a factor of TILE.
#define TILE 16
__global__ void k_matmul_tiled(const float *A, const float *B, float *C, int M,
                               int K, int N) {
  __shared__ float As[TILE][TILE];
  __shared__ float Bs[TILE][TILE];

  int tx = threadIdx.x, ty = threadIdx.y;
  int row = blockIdx.y * TILE + ty;
  int col = blockIdx.x * TILE + tx;
  float acc = 0.0f;

  for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
    int a_col = t * TILE + tx;
    int b_row = t * TILE + ty;

    // out-of-range loads become 0 so partial edge tiles still work
    As[ty][tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[ty][tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
    __syncthreads(); // tile fully loaded before anyone reads it

    for (int k = 0; k < TILE; k++)
      acc += As[ty][k] * Bs[k][tx];
    __syncthreads(); // everyone done reading before the next load overwrites
  }

  if (row < M && col < N)
    C[row * N + col] = acc;
}

// Block-level tree reduction in shared memory, then one atomicAdd per block.
__global__ void k_sum_all(const float *a, float *out, int n) {
  __shared__ float s[THREADS];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  s[tid] = i < n ? a[i] : 0.0f;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      s[tid] += s[tid + stride];
    __syncthreads();
  }

  if (tid == 0)
    atomicAdd(out, s[0]);
}

// ---------------------------------------------------------------- host API

Matrix *matrix_create(Arena *arena, int rows, int cols) {
  Matrix *mat = PUSH_STRUCT(arena, Matrix); // header: host memory
  mat->rows = rows;
  mat->cols = cols;
  mat->size = rows * cols;
  mat->data = (float *)arena_push_data(arena, sizeof(float) * (ull)rows *
                                                  cols); // GPU memory

  return mat;
}

Matrix *matrix_relu(Arena *arena, Matrix *a) {
  Matrix *result = matrix_create(arena, a->rows, a->cols);
  k_relu<<<blocks_for(a->size), THREADS>>>(a->data, result->data, a->size);
  KERNEL_CHECK();
  return result;
}

double power(double base, long long exp) {
  if (exp == 0)
    return 1.0;

  bool isNegative = exp < 0;
  ull exponent = isNegative ? -exp : exp;
  double result = 1.0;

  while (exponent > 0) {
    if (exponent & 1)
      result *= base;
    base *= base;
    exponent >>= 1;
  }

  return isNegative ? 1 / result : result;
}

Matrix *matrix_softmax(Arena *arena, Matrix *a) {
  Matrix *result = matrix_create(arena, a->rows, a->cols);
  k_softmax<<<blocks_for(a->rows), THREADS>>>(a->data, result->data, a->rows,
                                              a->cols);
  KERNEL_CHECK();
  return result;
}

Matrix *matrix_cross_entropy(Arena *arena, const Matrix *p, const Matrix *q) {
  if (p->rows != q->rows || p->cols != q->cols) {
    cout << "Matrix Cross Entropy: Mismatch of matrix dimensions, exiting...";
    return nullptr;
  }

  Matrix *result = matrix_create(arena, p->rows, p->cols);
  k_cross_entropy<<<blocks_for(p->size), THREADS>>>(p->data, q->data,
                                                    result->data, p->size);
  KERNEL_CHECK();
  return result;
}

Matrix *matrix_add_bias(Arena *arena, Matrix *input, Matrix *bias) {
  if (input->cols != bias->cols)
    return nullptr;

  Matrix *result = matrix_create(arena, input->rows, input->cols);
  k_add_bias<<<blocks_for(input->size), THREADS>>>(
      input->data, bias->data, result->data, input->size, input->cols);
  KERNEL_CHECK();
  return result;
}

Matrix *matrix_sum(Arena *arena, Matrix *a, Matrix *b) {
  if (a->rows != b->rows || a->cols != b->cols)
    return nullptr;

  Matrix *result = matrix_create(arena, a->rows, a->cols);
  k_add<<<blocks_for(a->size), THREADS>>>(a->data, b->data, result->data,
                                          a->size);
  KERNEL_CHECK();
  return result;
}

Matrix *matrix_transpose(Arena *arena, Matrix *a) {
  Matrix *result = matrix_create(arena, a->cols, a->rows);
  dim3 block(16, 16);
  dim3 grid((a->cols + 15) / 16, (a->rows + 15) / 16);
  k_transpose<<<grid, block>>>(a->data, result->data, a->rows, a->cols);
  KERNEL_CHECK();
  return result;
}

Matrix *matrix_mult(Arena *arena, Matrix *a, Matrix *b) {
  if (a->cols != b->rows)
    return nullptr;

  Matrix *result = matrix_create(arena, a->rows, b->cols);
  dim3 block(TILE, TILE);
  // grid x covers the columns of C, grid y covers the rows of C
  dim3 grid((b->cols + TILE - 1) / TILE, (a->rows + TILE - 1) / TILE);

#ifdef NAIVE_MATMUL
  k_matmul_naive<<<grid, block>>>(a->data, b->data, result->data, a->rows,
                                  a->cols, b->cols);
#else
  k_matmul_tiled<<<grid, block>>>(a->data, b->data, result->data, a->rows,
                                  a->cols, b->cols);
#endif
  KERNEL_CHECK();
  return result;
}

Matrix *matrix_load(Arena *arena, ull rows, ull cols, const string filepath) {
  Matrix *mat = matrix_create(arena, rows, cols);

  ifstream matfile(filepath, ios::binary);

  if (!matfile.is_open()) {
    cout << "Failed to open file: " << filepath << " , exiting...";
    exit(1);
  }

  // Stream the file to the GPU through a 1 MiB host buffer taken from the
  // arena's host side, then pop it. pos is saved *after* matrix_create, so the
  // pop only frees the staging buffer.
  const ull CHUNK = 256 * 1024; // floats
  ull saved_pos = arena->pos;
  float *buffer = PUSH_ARRAY(arena, float, CHUNK);

  ull total = rows * cols;
  for (ull offset = 0; offset < total; offset += CHUNK) {
    matfile.read((char *)buffer, sizeof(float) * min(CHUNK, total - offset));
    ull got = (ull)matfile.gcount() / sizeof(float);
    if (got == 0)
      break; // short file: the rest stays zero
    CUDA_CHECK(cudaMemcpy(mat->data + offset, buffer, sizeof(float) * got,
                          cudaMemcpyHostToDevice));
  }

  arena_pop_to(arena, saved_pos);
  matfile.close();

  return mat;
}

Matrix *matrix_relu_grad(Arena *arena, Matrix *layer_input,
                         Matrix *grad_output) {
  if (layer_input->rows != grad_output->rows ||
      layer_input->cols != grad_output->cols) {
    cout << "Matrix ReLU Grad: Mismatch of matrix dimensions, exiting...";
    return nullptr;
  }

  Matrix *result = matrix_create(arena, grad_output->rows, grad_output->cols);
  k_relu_grad<<<blocks_for(result->size), THREADS>>>(
      layer_input->data, grad_output->data, result->data, result->size);
  KERNEL_CHECK();
  return result;
}

Matrix *matrix_cross_entropy_grad(Arena *arena, Matrix *predictions,
                                  Matrix *labels) {
  if (predictions->rows != labels->rows || predictions->cols != labels->cols) {
    cout << "Matrix Cross Entropy Grad: Mismatch of matrix dimensions, "
            "exiting...";
    return nullptr;
  }

  Matrix *result = matrix_create(arena, predictions->rows, predictions->cols);
  k_sub<<<blocks_for(result->size), THREADS>>>(predictions->data, labels->data,
                                               result->data, result->size);
  KERNEL_CHECK();
  return result;
}

Matrix *matrix_sum_rows(Arena *arena, Matrix *a) {
  Matrix *result = matrix_create(arena, 1, a->cols);
  k_sum_rows<<<blocks_for(a->cols), THREADS>>>(a->data, result->data, a->rows,
                                               a->cols);
  KERNEL_CHECK();
  return result;
}

void matrix_clip_inplace(Matrix *a, float max_abs) {
  k_clip<<<blocks_for(a->size), THREADS>>>(a->data, max_abs, a->size);
  KERNEL_CHECK();
}

void matrix_copy_to_host(const Matrix *m, float *host_dst) {
  CUDA_CHECK(cudaMemcpy(host_dst, m->data, sizeof(float) * (ull)m->size,
                        cudaMemcpyDeviceToHost));
}

void matrix_copy_from_host(Matrix *m, const float *host_src) {
  CUDA_CHECK(cudaMemcpy(m->data, host_src, sizeof(float) * (ull)m->size,
                        cudaMemcpyHostToDevice));
}

void matrix_copy_rows(Matrix *dst, const Matrix *src, int start_row) {
  // rows are contiguous, so a batch is one GPU-to-GPU block copy
  CUDA_CHECK(cudaMemcpy(dst->data, src->data + (ull)start_row * src->cols,
                        sizeof(float) * (ull)dst->size,
                        cudaMemcpyDeviceToDevice));
}

float matrix_sum_all(Arena *arena, const Matrix *a) {
  // one-float accumulator from the scratch arena (arena_push_data zeroes it)
  float *d_sum = (float *)arena_push_data(arena, sizeof(float));
  k_sum_all<<<blocks_for(a->size), THREADS>>>(a->data, d_sum, a->size);
  KERNEL_CHECK();

  float sum;
  // blocking copy: waits for all queued GPU work to finish (a sync point)
  CUDA_CHECK(cudaMemcpy(&sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
  return sum;
}

void matrix_scale_inplace(Matrix *a, float s) {
  k_scale<<<blocks_for(a->size), THREADS>>>(a->data, s, a->size);
  KERNEL_CHECK();
}

void matrix_sgd_update(Matrix *w, const Matrix *dw, float lr) {
  k_sgd<<<blocks_for(w->size), THREADS>>>(w->data, dw->data, lr, w->size);
  KERNEL_CHECK();
}
