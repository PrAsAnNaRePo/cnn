#ifndef PRETOK_H
#define PRETOK_H
#include <stddef.h>

typedef struct { const unsigned char *s; size_t n; } Src;
size_t pretok_next(const Src *t, size_t i);

/* Equivalent of Python: token_pattern.findall(text) */
typedef struct {
    char  **items;   /* items[k] is a NUL-terminated copy of the k-th substring */
    size_t *lens;    /* lens[k] is its length in bytes (excluding the NUL) */
    size_t  count;   /* number of substrings */
} Chunks;

Chunks pretok_split(const char *text, size_t n);
void   chunks_free(Chunks *c);

#endif
