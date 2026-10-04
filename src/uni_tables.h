#ifndef UNI_TABLES_H
#define UNI_TABLES_H
#include <stdint.h>
#include <stddef.h>
typedef struct { uint32_t lo, hi; uint8_t f; } Range;
extern const unsigned char ASCII_FLAGS[128];
extern const Range UNI_RANGES[];
extern const size_t UNI_NRANGES;
enum { F_U = 1, F_W = 2, F_L = 4, F_N = 8, F_S = 16 };
#endif
