#pragma once

#include <bits/stdc++.h>
using namespace std;

#define KiB(n) ((unsigned long long)n << 10)
#define MiB(n) ((unsigned long long)n << 20)
#define GiB(n) ((unsigned long long)n << 30)

#define ARENA_BASE_POS (sizeof(Arena))
#define ARENA_ALIGN (sizeof(void *))
#define ARENA_DEVICE_ALIGN 256ull // GPU buffers aligned for coalesced access

#define PUSH_STRUCT(arena, T) (T *)arena_push((arena), sizeof(T))
#define PUSH_ARRAY(arena, T, n) (T *)arena_push((arena), sizeof(T) * (n))

typedef unsigned long long ull;

// plain function instead of a C++20 concept so nvcc can build with -std=c++17
inline ull align_up(ull value, ull power) {
  return (value + (power - 1)) & ~(power - 1);
}

class Arena {
public:
  ull capacity;
  ull pos;
#ifdef USE_CUDA
  // second bump allocator over one cudaMalloc'd block: matrix headers stay
  // in host memory (above), matrix data comes from here
  char *dev_base;
  ull dev_capacity;
  ull dev_pos;
#endif
};

Arena *arena_create(ull capacity);
void arena_destroy(Arena *arena);
void *arena_push(Arena *arena, ull size);
// Memory for matrix data: host memory in the CPU build, GPU memory in the
// CUDA build. PUSH_STRUCT / PUSH_ARRAY always return host memory.
void *arena_push_data(Arena *arena, ull size);
// pop / pop_to only move the host side; use them for host temporaries only.
void arena_pop(Arena *arena, ull size);
void arena_pop_to(Arena *arena, ull pos);
// clear resets both sides
void arena_clear(Arena *arena);
