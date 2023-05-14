import os
import numpy as np
import math

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from flax.linen.dtypes import promote_dtype

from typing import Optional, Any
Dtype = Any


def scaled_dot_attention(query : jnp.ndarray,
                         key : jnp.ndarray,
                         value : jnp.ndarray,
                         mask : Optional[jnp.ndarray] = None,
                         dtype : Optional[Dtype] = None
                       ):


    """
    Compute scaled attention given query, key and value from the paper 'Attention is All You Need'
    This method is for multi-head attention

    Args :

        query : b x ... x num_heads x seq_len x d_k_per_head
        key   : b x ... x num_heads x seq_len x d_k_per_head
        value : b x ... x num_heads x seq_len x d_k_per_head
        mask  : b x ... x num_heads x seq_len x d_k_per_head

    """

    # for q, k and v check the number of dimensions.

    assert key.ndim == query.ndim == value.ndim, 'query, key and value must have same rank'
    assert query.shape[:-3] == key.shape[:-3] == value.shape[:-3], 'query, key and value batch dimensions must match'
    assert query.shape[-3] == key.shape[-3] == value.shape[-3], 'query, key and value must have same number of heads (-2 dimension)'
    assert query.shape[-2] == key.shape[-2] == value.shape[-2], 'query, key and value must have the same sequence length dimension (-3 dimension)'
    assert query.shape[-1] == key.shape[-1] == value.shape[-1], 'query, key and value must have the same token dimension d_k (-1 dimension)'

    # make sure that q, k and v are can be broadcasted to the smallest common dtype
    query, key, value = promote_dtype(query, key, value, dtype = dtype)
    dtype = query.dtype

    d_k = query.shape[-1]
    attn_logits = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) # Q.K^T, b x seq_len x num_heads x num_heads

    attn_logits = attn_logits / jnp.sqrt(d_k)
    if mask is not None:
        attn_logits = jnp.where(mask ==0, -9e15, attn_logits)

    attention = nn.softmax(attn_logits, axis = -1) # weights for the values
    values = jnp.matmul(attention, value) # b x seq_len x num_heads x d_k

    return values




class MultiheadAttention(nn.Module):

    embed_dim : int
    num_heads : int

    """

    Below setup() method is used. It allows pytorch style forward pass.
    It's alternative nn.compact is analogous to keras.

    setup() allows more than one forward pass method
    (https://flax.readthedocs.io/en/latest/guides/setup_or_nncompact.html)

    """

    def setup(self):

        self.qkv_proj = nn.Dense(3 * self.embed_dim,
                                 kernel_init = nn.initializers.xavier_uniform(),
                                 bias_init = nn.initializers.zeros)

        self.out_proj = nn.Dense(self.embed_dim,
                                 kernel_init = nn.initializers.xavier_uniform(),
                                 bias_init = nn.initializers.zeros)

    def __call__(self, x, mask = None):

        batch_size, seq_len, embed_dim = x.shape
        assert embed_dim % self.num_heads == 0, "embed_dim modulo 0 should be num_heads"

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, -1)
        qkv = qkv.transpose(0, 2, 1, 3)
        query, key, value = jnp.array_split(qkv, 3, axis = -1)

        values = scaled_dot_attention(query, key, value, mask = mask)
        values = values.transpose(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, embed_dim)
        out = self.out_proj(values)


        return out


# a single encoder block for transformer

class EncoderBlock(nn.Module):

    input_dim : int # it is equal to output dim
    num_heads : int
    dim_feedforward : int
    dropout_prob : float


    def setup(self):

        self.mh_att = MultiheadAttention(embed_dim = self.input_dim,
                                         num_heads = self.num_heads)

        self.linear_layers = [nn.Dense(self.dim_feedforward),
                              nn.Dropout(self.dropout_prob),
                              nn.relu,
                              nn.Dense(self.input_dim)]

        self.layer_norm1 = nn.LayerNorm()
        self.layer_norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(self.dropout_prob)


    def __call__(self, x, mask = None, train = True):

        attn_out = self.mh_att(x, mask = None)
        x =  x + self.dropout(attn_out, deterministic = not train)
        x = self.layer_norm1(x)

        # mlp aprt
        linear_out = x # x is needed for residual connection later for residual connection
        for l in self.linear_layers:
            if isinstance(l, nn.Dropout):
                linear_out = l(linear_out, deterministic = not train)
            else :
                linear_out = l(linear_out)
        x = x + self.dropout(linear_out, deterministic = not train)
        x = self.layer_norm2(x)

        return x









