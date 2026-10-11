#include "arena.hpp"

#ifdef USE_CUDA
#include <cuda_runtime.h>
// in the CUDA build the host side only holds Matrix/Layer headers and small
// host staging buffers, so it does not need the full capacity
static const ull HOST_CAPACITY = MiB(8);
#endif

Arena *arena_create(ull capacity) {
#ifdef USE_CUDA
  ull host_capacity = HOST_CAPACITY;
#else
  ull host_capacity = capacity;
#endif

  Arena *arena = (Arena *)malloc(host_capacity);

  arena->capacity = host_capacity;
  arena->pos = ARENA_BASE_POS;

#ifdef USE_CUDA
  if (cudaMalloc((void **)&arena->dev_base, capacity) != cudaSuccess) {
    cout << "cudaMalloc of " << (capacity >> 20) << " MiB failed, exiting...";
    exit(1);
  }
  arena->dev_capacity = capacity;
  arena->dev_pos = 0;
#endif

  return arena;
}

void arena_destroy(Arena *arena) {
#ifdef USE_CUDA
  cudaFree(arena->dev_base);
#endif
  free(arena);
}

void *arena_push(Arena *arena, ull size) {
  ull aligned_pos = align_up(arena->pos, (ull)ARENA_ALIGN);
  ull new_pos = aligned_pos + size;

  if (new_pos > arena->capacity) {
    cout << "Program tried to allocate more memory than was available, "
            "terminating...";
    return nullptr;
  }

  arena->pos = new_pos;

  char *outptr = (char *)arena + aligned_pos;
  memset(outptr, 0, size);
  return outptr;
}

void *arena_push_data(Arena *arena, ull size) {
#ifdef USE_CUDA
  ull aligned_pos = align_up(arena->dev_pos, ARENA_DEVICE_ALIGN);
  ull new_pos = aligned_pos + size;

  if (new_pos > arena->dev_capacity) {
    cout << "Program tried to allocate more GPU memory than was available, "
            "terminating...";
    return nullptr;
  }

  arena->dev_pos = new_pos;

  char *outptr = arena->dev_base + aligned_pos;
  cudaMemset(outptr, 0, size); // ordered on the default stream
  return outptr;
#else
  return arena_push(arena, size);
#endif
}

void arena_pop(Arena *arena, ull size) {
  size = min(size, arena->pos - ARENA_BASE_POS);
  arena->pos -= size;
}

void arena_pop_to(Arena *arena, ull pos) {
  ull size = pos < arena->pos ? arena->pos - pos : 0;
  arena_pop(arena, size);
}

void arena_clear(Arena *arena) {
  arena_pop_to(arena, ARENA_BASE_POS);
#ifdef USE_CUDA
  arena->dev_pos = 0;
#endif
}
