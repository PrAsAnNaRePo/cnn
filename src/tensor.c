#include "tensor.h"
#include "arena.h"
#include "config.h"
#include "types.h"
#include <stdlib.h>
#include <string.h>

Tensor *create_tensor(Arena *arena, uint32 *shape, uint32 num_dim,
                      WEI_TYPE wei_init, uint8 trainable) {
  Tensor *tensor = (Tensor *)arena_alloc(arena, sizeof(Tensor));

  tensor->shape = (uint32 *)arena_alloc(arena, sizeof(uint32) * num_dim);
  for (uint32 i = 0; i < num_dim; i++) {
    tensor->shape[i] = shape[i];
  }

  tensor->num_dim = num_dim;

  tensor->numel = tensor->shape[0];
  for (uint32 i = 1; i < num_dim; i++) {
    tensor->numel *= tensor->shape[i];
  }

  tensor->data = (WEI_TYPE *)arena_alloc(arena, sizeof(WEI_TYPE) * tensor->numel);
  for (uint32 i = 0; i < tensor->numel; i++) {
    tensor->data[i] = wei_init;
  }
  tensor->grad = (WEI_TYPE *)arena_alloc(arena, sizeof(WEI_TYPE) * tensor->numel);
  for (uint32 i = 0; i < tensor->numel; i++) {
    tensor->grad[i] = 0.0;
  }
  tensor->grad_fn = NULL;
  
  if (trainable) {
    tensor->m = (WEI_TYPE *)arena_alloc(arena, sizeof(WEI_TYPE) * tensor->numel);
    tensor->v = (WEI_TYPE *)arena_alloc(arena, sizeof(WEI_TYPE) * tensor->numel);
    for (uint32 i = 0; i < tensor->numel; i++) {
      tensor->m[i] = 0.0;
      tensor->v[i] = 0.0;
    }
  } else {
    tensor->m = NULL;
    tensor->v = NULL;
  }

  return tensor;
}

void print_tensor(const Tensor *tensor) {
  if (!tensor) {
    printf("Tensor: NULL\n");
    return;
  }
  printf("Tensor (shape=[");
  for (uint32 i = 0; i < tensor->num_dim; i++) {
    printf("%u%s", tensor->shape[i], (i + 1 < tensor->num_dim) ? ", " : "");
  }
  printf("], numel=%u):\n  [", tensor->numel);
  for (uint32 i = 0; i < tensor->numel; i++) {
    printf("%.4f%s", tensor->data[i], (i + 1 < tensor->numel) ? ", " : "");
  }
  printf("]\n");
}

void print_shape(const Tensor *tensor) {
  if (!tensor) {
    printf("Tensor: NULL\n");
    return;
  } else {
    printf("Tensor (shape=[");
    for (uint32 i = 0; i < tensor->num_dim; i++) {
      printf("%u%s", tensor->shape[i], (i + 1 < tensor->num_dim) ? ", " : "");
    }
    printf("], numel=%u)\n", tensor->numel);
  }
}

void print_tensor_grad(const Tensor *tensor) {
  if (!tensor || !tensor->grad) {
    printf("Tensor Grad: NULL\n");
    return;
  }
  printf("Tensor Grad (shape=[");
  for (uint32 i = 0; i < tensor->num_dim; i++) {
    printf("%u%s", tensor->shape[i], (i + 1 < tensor->num_dim) ? ", " : "");
  }
  printf("], numel=%u):\n  [", tensor->numel);
  for (uint32 i = 0; i < tensor->numel; i++) {
    printf("%.4f%s", tensor->grad[i], (i + 1 < tensor->numel) ? ", " : "");
  }
  printf("]\n");
}

Tensor *add_tensor(Arena *arena, Tensor *tensor1, Tensor *tensor2) {
  if (tensor1->numel != tensor2->numel) {
    return NULL;
  }
  for (uint32 i = 0; i < tensor1->num_dim; i++) {
    if (tensor1->shape[i] != tensor2->shape[i])
      return NULL;
  }

  Tensor *c = create_tensor(arena, tensor1->shape, tensor1->num_dim, 0.0, 0);

  for (uint32 i = 0; i < tensor1->numel; i++) {
    c->data[i] = tensor1->data[i] + tensor2->data[i];
  }

  c->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  c->grad_fn->ops = ADD;
  c->grad_fn->num_inputs = 2;
  c->grad_fn->inputs[0] = tensor1;
  c->grad_fn->inputs[1] = tensor2;
  c->grad_fn->output = c;
  c->grad_fn->visited = 0;

  return c;
}

Tensor *sub_tensor(Arena *arena, Tensor *tensor1, Tensor *tensor2) {
  if (tensor1->numel != tensor2->numel) {
    return NULL;
  }
  for (uint32 i = 0; i < tensor1->num_dim; i++) {
    if (tensor1->shape[i] != tensor2->shape[i])
      return NULL;
  }

  Tensor *c = create_tensor(arena, tensor1->shape, tensor1->num_dim, 0.0, 0);

  for (uint32 i = 0; i < tensor1->numel; i++) {
    c->data[i] = tensor1->data[i] - tensor2->data[i];
  }

  c->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  c->grad_fn->ops = SUB;
  c->grad_fn->num_inputs = 2;
  c->grad_fn->inputs[0] = tensor1;
  c->grad_fn->inputs[1] = tensor2;
  c->grad_fn->output = c;
  c->grad_fn->visited = 0;

  return c;
}

// Tensor *mul_tensor(Arena *arena, Tensor *tensor1, Tensor *tensor2){
//   if (!tensor1 || !tensor2) return NULL;
//   if (tensor1->num_dim != tensor2->num_dim) return NULL;
//   if (tensor1->shape[tensor1->num_dim - 1] != tensor2->shape[tensor2->num_dim
//   - 2]) return NULL;
//
//   uint32 shape[tensor1->num_dim];
//   uint32 total_batch = 1;
//   for (int i = 0; i < tensor1->num_dim; i++){
//     if (i < tensor1->num_dim - 2) {
//       if (tensor1->shape[i] != tensor2->shape[i]) return NULL;
//       total_batch = total_batch * tensor1->shape[i];
//     }
//     shape[i] = tensor1->shape[i];
//   }
//   shape[tensor1->num_dim - 1] = tensor2->shape[tensor2->num_dim - 1];
//
//   Tensor *c = create_tensor(arena, shape, tensor1->num_dim, 0.0);
//   if (!c) return NULL;
//
//   for (uint32 b = 0; b < total_batch; b++){
//     uint32 s_batch_a = b * tensor1->shape[tensor1->num_dim-2] *
//     tensor1->shape[tensor1->num_dim-1]; uint32 s_batch_b = b *
//     tensor2->shape[tensor2->num_dim-2] * tensor2->shape[tensor2->num_dim-1];
//     uint32 s_batch_c = b * c->shape[c->num_dim-2]*c->shape[c->num_dim-1];
//
//     for (uint32 i = 0; i < c->shape[tensor1->num_dim-2]; i++){
//       uint32 s_row_a = tensor1->shape[tensor1->num_dim-1]*i;
//       for (uint32 j = 0; j < c->shape[tensor1->num_dim-1]; j++){
//         fp32 cum_add = 0.0f;
//         for (uint32 k = 0; k < tensor1->shape[tensor1->num_dim-1]; k++){
//           cum_add += tensor1->data[s_batch_a + s_row_a + k] *
//           tensor2->data[s_batch_b + tensor2->shape[tensor2->num_dim-1]*k +
//           j];
//         }
//         c->data[s_batch_c + c->shape[c->num_dim-1]*i + j] = cum_add;
//       }
//     }
//   }
//   return c;
// }

Tensor *mul_tensor(Arena *arena, Tensor *tensor1, Tensor *tensor2) {
  if (!tensor1 || !tensor2)
    return NULL;
  if (tensor1->num_dim != tensor2->num_dim)
    return NULL;

  int ndim = tensor1->num_dim;
  if (ndim < 2)
    return NULL;

  if (tensor1->shape[ndim - 1] != tensor2->shape[ndim - 2])
    return NULL;

  uint32 shape[16];
  uint32 total_batch = 1;
  for (int i = 0; i < ndim; i++) {
    if (i < ndim - 2) {
      if (tensor1->shape[i] != tensor2->shape[i])
        return NULL;
      total_batch *= tensor1->shape[i];
    }
    shape[i] = tensor1->shape[i];
  }
  shape[ndim - 1] = tensor2->shape[ndim - 1];

  Tensor *c = create_tensor(arena, shape, ndim, 0.0f, 0);
  if (!c)
    return NULL;

  size_t M = tensor1->shape[ndim - 2];
  size_t K = tensor1->shape[ndim - 1];
  size_t N = tensor2->shape[ndim - 1];

  for (size_t b = 0; b < total_batch; b++) {
    size_t s_batch_a = b * M * K;
    size_t s_batch_b = b * K * N;
    size_t s_batch_c = b * M * N;

    for (size_t i = 0; i < M; i++) {
      size_t s_row_a = K * i;
      size_t s_row_c = N * i;

      for (size_t j = 0; j < N; j++) {
        WEI_TYPE cum_add = 0.0f;
        for (size_t k = 0; k < K; k++) {
          cum_add += tensor1->data[s_batch_a + s_row_a + k] *
                     tensor2->data[s_batch_b + N * k + j];
        }
        c->data[s_batch_c + s_row_c + j] = cum_add;
      }
    }
  }
  c->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  c->grad_fn->ops = MATMUL;
  c->grad_fn->num_inputs = 2;
  c->grad_fn->inputs[0] = tensor1;
  c->grad_fn->inputs[1] = tensor2;
  c->grad_fn->output = c;
  c->grad_fn->visited = 0;
  return c;
}

WEI_TYPE sum_tensor(Tensor *tensor) {
  WEI_TYPE sum = 0.0f;
  for (uint32 i = 0; i < tensor->numel; i++) {
    sum += tensor->data[i];
  }
  return sum;
}

Tensor *scale_tensor(Arena *arena, Tensor *tensor, WEI_TYPE scalar){

  Tensor *out = create_tensor(arena, tensor->shape, tensor->num_dim, 0.0f, 0);
  if (!out)
    return NULL;

  for (uint32 i = 0; i < tensor->numel; i++) {
    out->data[i] = scalar * tensor->data[i];
  }

  out->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  out->grad_fn->ops = SCALE;
  out->grad_fn->num_inputs = 1;
  out->grad_fn->inputs[0] = tensor;
  out->grad_fn->output = out;
  out->grad_fn->visited = 0;
  out->grad_fn->scalar = scalar;
  return out;
}

Tensor *copy_tensor(Arena *arena, const Tensor *tensor) {
  if (!tensor)
    return NULL;

  Tensor *dst = create_tensor(arena, tensor->shape, tensor->num_dim, 0.0f, 0);
  if (!dst)
    return NULL;

  if (tensor->data && dst->data) {
    memcpy(dst->data, tensor->data, sizeof(WEI_TYPE) * tensor->numel);
  }
  if (tensor->grad && dst->grad) {
    memcpy(dst->grad, tensor->grad, sizeof(WEI_TYPE) * tensor->numel);
  }

  return dst;
}

Tensor *reshape_tensor(Arena *arena, Tensor *tensor, uint32 *dshape,
                       uint32 dnum_dim) {
  if (!tensor)
    return NULL;

  uint32 new_numel = 1;
  for (uint32 i = 0; i < dnum_dim; i++) {
    new_numel *= dshape[i];
  }
  if (new_numel != tensor->numel)
    return NULL;

  Tensor *dst = (Tensor *)arena_alloc(arena, sizeof(Tensor));
  if (!dst)
    return NULL;
  dst->data = tensor->data;
  dst->grad = tensor->grad;
  dst->numel = tensor->numel;

  dst->shape = (uint32 *)arena_alloc(arena, sizeof(uint32) * dnum_dim);
  for (uint32 i = 0; i < dnum_dim; i++) {
    dst->shape[i] = dshape[i];
  }
  dst->num_dim = dnum_dim;
  dst->grad_fn = tensor->grad_fn;

  return dst;
}

Tensor *transpose_tensor(Arena *arena, Tensor *tensor, uint32 ax1, uint32 ax2){
  
  if (!tensor) return NULL;
  if (ax1 == ax2 || ax1 >= tensor->num_dim || ax2 >= tensor->num_dim) return NULL;

  uint32 dshape[tensor->num_dim];
  for (uint32 i = 0; i < tensor->num_dim; i++){
    dshape[i] = tensor->shape[i];
  }
  dshape[ax1] = tensor->shape[ax2];
  dshape[ax2] = tensor->shape[ax1];

  Tensor *dst = create_tensor(arena, dshape, tensor->num_dim, 0.0f, 0);
  if (!dst) return NULL;

  uint32 src_multipliers[tensor->num_dim], dst_multipliers[tensor->num_dim];

  for (uint32 i = 0; i < tensor->num_dim; i++){
    src_multipliers[i] = 1;
    dst_multipliers[i] = 1;
    for (uint32 j = i + 1; j < tensor->num_dim; j++){
      src_multipliers[i] *= tensor->shape[j];
      dst_multipliers[i] *= dshape[j];
    }
  }

  uint32 coords[tensor->num_dim];
  for (uint32 i = 0; i < tensor->numel; i++){
    uint32 left = i;
    for (uint32 j = 0; j < tensor->num_dim; j++){
      coords[j] = left / src_multipliers[j];
      left = left % src_multipliers[j];
    }

    uint32 tmp = coords[ax1];
    coords[ax1] = coords[ax2];
    coords[ax2] = tmp;

    uint32 idx = 0;
    for (uint32 j = 0; j < tensor->num_dim; j++){
      idx += coords[j] * dst_multipliers[j];
    }
    dst->data[idx] = tensor->data[i];
  }

  dst->grad_fn = (Node *)arena_alloc(arena, sizeof(Node));
  dst->grad_fn->ops = TRANSPOSE;
  dst->grad_fn->num_inputs = 1;
  dst->grad_fn->inputs[0] = tensor;
  dst->grad_fn->output = dst;
  dst->grad_fn->visited = 0;
  dst->grad_fn->axis1 = ax1;
  dst->grad_fn->axis2 = ax2;

  return dst;
}

