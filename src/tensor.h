#pragma once

#include "types.h"
#include "config.h"
#include "arena.h"

typedef struct Node Node;

typedef struct Tensor {
    uint32 *shape;
    uint32 num_dim;
    WEI_TYPE *data, *grad;

    uint32 *strides;
    uint32 numel;
    Node* grad_fn;
} Tensor;

enum OPS{
    ADD,
    MATMUL,
    SUB,

    RELU,
    SIGMOID,
    SOFTMAX,

    MSE_LOSS,
    SOFTMAX_CROSS_ENTROPY
};

struct Node{
    enum OPS ops;
    uint32 num_inputs;
    Tensor* inputs[3]; 
    Tensor* output;
    uint32 visited;
};

Tensor *create_tensor(Arena *arena, uint32 *shape, uint32 num_dim, WEI_TYPE wei_init);
void print_tensor(const Tensor *tensor);
void print_tensor_grad(const Tensor *tensor);

Tensor *add_tensor(Arena *arena, Tensor *tensor1, Tensor *tensor2);
Tensor *sub_tensor(Arena *arena, Tensor *tensor1, Tensor *tensor2);
Tensor *mul_tensor(Arena *arena, Tensor *tensor1, Tensor *tensor2);
WEI_TYPE sum_tensor(Tensor *tensor);
void scale_tensor(Tensor *tensor, WEI_TYPE scalar);
Tensor *copy_tensor(Arena *arena, const Tensor *tensor);
Tensor *reshape_tensor(Arena *arena, Tensor *tensor, uint32 *dshape, uint32 dnum_dim);
