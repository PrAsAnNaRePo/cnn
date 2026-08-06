#include <stdlib.h>
#include "arena.h"

Arena arena_create(size_t capacity){
  Arena arena = {
    .base = (char *)malloc(capacity),
    .offset = 0,
    .capacity = capacity
  };
  return arena;
}

void *arena_alloc(Arena *arena, size_t size){
  arena->offset = ALIGN_OFFSET(arena->offset);
  if (arena->offset + size > arena->capacity){
    return NULL;
  }
  char *ptr = arena->base + arena->offset;
  arena->offset += size;
  return ptr;
}

void arena_clear(Arena *arena){
  arena->offset = 0;
}

void arena_free(Arena *arena){
  free(arena->base);

  arena->base = NULL;
  arena->offset = 0;
  arena->capacity = 0;
}
