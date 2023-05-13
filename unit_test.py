# testing modules independently
import numpy as np
import jax.numpy as jnp
from attention import scaled_dot_attention
from attention import MultiheadAttention
from jax import random

# testing scaled dot attention for multi head sqeuences

q = np.random.rand(10,10, 5, 5)
k = np.random.rand(10, 10, 5, 5)
v = np.random.rand(10, 10, 5, 5)

q = jnp.asarray(q)
k = jnp.asarray(k)
v = jnp.asarray(v)

ans = scaled_dot_attention(q, k, v)
print(ans.shape)
if ans.shape == (10,10,5,5):
    print("scaled dot attention test passed")



# testing multihead attention module

main_rng = random.PRNGKey(seed = 42)
main_rng, x_rng = random.split(main_rng)
x = np.random.rand(4, 16, 128)
x = jnp.asarray(x)
params = mh_attn.init(init_rng, x)['params']
mh_attn = MultiheadAttention(embed_dim = 128, num_heads = 4)
out = mh_attn.apply(x)
