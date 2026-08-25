#include <bits/stdc++.h>
using namespace std;

#define KiB(n) ((unsigned long long)n << 10)
#define MiB(n) ((unsigned long long)n << 20)
#define GiB(n) ((unsigned long long)n << 30)

#define ARENA_BASE_POS (sizeof(Arena))
#define ARENA_ALIGN (sizeof(void *))

#define PUSH_STRUCT(arena, T) (T *)arena_push((arena), sizeof(T))
#define PUSH_ARRAY(arena, T, n) (T *)arena_push((arena), sizeof(T) * (n))

typedef unsigned long long ull;

template <unsigned_integral T> constexpr T align_up(T value, T power) noexcept {
  return (value + (power - 1)) & ~(power - 1);
}

class Arena {
public:
  ull capacity;
  ull pos;
};

Arena *arena_create(ull capacity) {
  Arena *arena = (Arena *)malloc(capacity);

  arena->capacity = capacity;
  arena->pos = ARENA_BASE_POS;

  return arena;
}

void arena_destroy(Arena *arena) { free(arena); }

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

void arena_pop(Arena *arena, ull size) {
  size = min(size, arena->pos - ARENA_BASE_POS);
  arena->pos -= size;
}

void arena_pop_to(Arena *arena, ull pos) {
  ull size = pos < arena->pos ? arena->pos - pos : 0;
  arena_pop(arena, size);
}

void arena_clear(Arena *arena) { arena_pop_to(arena, ARENA_BASE_POS); }
