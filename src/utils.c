#include "arena.h"
#include "types.h"
#include "tensor.h"

WEI_TYPE mean(WEI_TYPE *arr, size_t size){
  WEI_TYPE sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += arr[i];
  }
  return sum / size;
}

WEI_TYPE var(WEI_TYPE *arr, WEI_TYPE mean, size_t size){
  WEI_TYPE sum = 0;
  for (size_t i = 0; i < size; i++) {
    sum += (arr[i] - mean) * (arr[i] - mean);
  }
  return sum / size;
}

Tensor *split_tensor(Arena *arena, Tensor *src, uint32 off, uint32 width){
  // always splits at last dimension of the tensor
  if (!src) return NULL;
  if (off + width > src->shape[src->num_dim - 1]) return NULL;

  uint32 *shape = (uint32 *)arena_alloc(arena, sizeof(uint32) * src->num_dim);
  for (uint32 i = 0; i < src->num_dim - 1; i++) {
    shape[i] = src->shape[i];
  }
  shape[src->num_dim - 1] = width;
  Tensor *dst = create_tensor(arena, shape, src->num_dim, 0.0f);
  if (!dst) return NULL;

  size_t bz = 1;
  for (uint32 i = 0; i < src->num_dim - 1; i++) {
    bz *= src->shape[i];
  }

  for (size_t i = 0; i < bz; i++) {
    for (uint32 j = 0; j < width; j++) {
      dst->data[i * width + j] = src->data[i * src->shape[src->num_dim - 1] + j + off];
    }
  }
  
  dst->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  dst->grad_fn->ops = SPLIT;
  dst->grad_fn->num_inputs = 1;
  dst->grad_fn->inputs[0] = src;
  dst->grad_fn->output = dst;
  dst->grad_fn->visited = 0;
  dst->grad_fn->offset = off;

  return dst;
}
