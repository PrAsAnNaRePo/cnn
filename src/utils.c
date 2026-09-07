#include "arena.h"
#include "types.h"
#include "tensor.h"

static uint32 rng_state = 0x9E3779B9;

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
  Tensor *dst = create_tensor(arena, shape, src->num_dim, 0.0f, 0);
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

Tensor *attention_mask_tensor(Arena *arena, Tensor *tensor){
  // expects input to be the shape of [B, h, s, s] 
  if (!tensor) return NULL;

  Tensor *mask = create_tensor(arena, tensor->shape, tensor->num_dim, -1e9f, 0);
  if (!mask) return NULL;
  
  uint32 bz = tensor->shape[0];
  uint32 H = tensor->shape[1];
  uint32 S = tensor->shape[2];

  for (uint32 b = 0; b < bz; b++){
    uint32 s_batch = b * H * S * S;
    for (uint32 h = 0; h < H; h++){
      uint32 h_row = h * S * S;
      for (uint32 i = 0; i < S; i++){
        for (uint32 j = 0; j < S; j++){
          if (j <= i) mask->data[s_batch + h_row + i * S + j] = tensor->data[s_batch + h_row + i * S + j];
        }
      }
    } 
  }

  mask->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  mask->grad_fn->ops = ATTN_MASK;
  mask->grad_fn->num_inputs = 1;
  mask->grad_fn->inputs[0] = tensor;
  mask->grad_fn->output = mask;
  mask->grad_fn->visited = 0;
  
  return mask;
}

uint32 xorshift32(void){
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

void init_uniform(Tensor *tensor, WEI_TYPE scale){
  for (uint32 i = 0; i < tensor->numel; i++){
    tensor->data[i] = ((WEI_TYPE)xorshift32() / (WEI_TYPE)UINT32_MAX * 2.0f - 1.0f) * scale; // [-scale, +scale]
  }
}
