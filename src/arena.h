#pragma once

#include <stddef.h>

typedef struct Arena {
  char* base;
  size_t offset;
  size_t capacity;
} Arena;

#define POINTER_SIZE sizeof(void*)
#define ALIGN_OFFSET(OFFSET) ((OFFSET) % (POINTER_SIZE) == 0 ? (OFFSET) : (OFFSET) + ((POINTER_SIZE) - ((OFFSET) % (POINTER_SIZE))))
#define KiB(x) ((size_t)(x) * 1024ULL)
#define MiB(x) ((size_t)(x) * 1024ULL * 1024ULL)
#define GiB(x) ((size_t)(x) * 1024ULL * 1024ULL * 1024ULL)

Arena arena_create(size_t capacity);
void *arena_alloc(Arena* arena, size_t size);
void arena_clear(Arena* arena);
void arena_free(Arena* arena);

