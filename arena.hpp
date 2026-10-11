#pragma once

#include<bits/stdc++.h>
using namespace std;

#define KiB(n) ((unsigned long long)n << 10)
#define MiB(n) ((unsigned long long)n << 20)
#define GiB(n) ((unsigned long long)n << 30)

#define ARENA_BASE_POS (sizeof(Arena))
#define ARENA_ALIGN (sizeof(void*))

#define PUSH_STRUCT(arena, T) (T*)arena_push((arena), sizeof(T))
#define PUSH_ARRAY(arena, T, n) (T*)arena_push((arena), sizeof(T) * (n))

typedef unsigned long long ull;

template <unsigned_integral T>
constexpr T align_up(T value, T power) noexcept {
    return (value + (power - 1)) & ~(power - 1);
}

class Arena {
    public:
    ull capacity;
    ull pos;
};

Arena* arena_create(ull capacity);
void arena_destroy(Arena* arena);
void* arena_push(Arena* arena, ull size);
// Memory for matrix data. Same as arena_push here; the CUDA build returns GPU
// memory instead. PUSH_STRUCT / PUSH_ARRAY always return host memory.
void* arena_push_data(Arena* arena, ull size);
void arena_pop(Arena* arena, ull size);
void arena_pop_to(Arena* arena, ull pos);
void arena_clear(Arena* arena);
