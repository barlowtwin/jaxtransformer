# testing modules independently
import numpy as np
import jax.numpy as jnp
from attention import scaled_dot_attention
from attention import MultiheadAttention
from jax import random
from attention import EncoderBlock
from attention import TransformerEncoder

# testing scaled dot attention for multi head sqeuences

q = np.random.rand(10,10, 5, 5)
k = np.random.rand(10, 10, 5, 5)
v = np.random.rand(10, 10, 5, 5)

q = jnp.asarray(q)
k = jnp.asarray(k)
v = jnp.asarray(v)

ans, attention = scaled_dot_attention(q, k, v)
if ans.shape == (10, 10, 5, 5):
    print("scaled dot attention test passed")
else :
    print("scaled dot attention not passed")


# testing multihead attention module

main_rng = random.PRNGKey(seed = 42)
main_rng, x_rng = random.split(main_rng)
main_rng, init_rng = random.split(main_rng)
x = np.random.rand(4, 16, 128)
x = jnp.asarray(x)
mh_attn = MultiheadAttention(embed_dim = 128, num_heads = 4)
params = mh_attn.init(init_rng, x)['params']
out, attention = mh_attn.apply({'params' : params} ,x)
if out.shape == (4, 16, 128):
    print("head attention passed")
else :
    print("multi head attention not passed")



# testing a single encoder block

main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (4, 16, 128))
encoder_block = EncoderBlock(input_dim = 128,num_heads = 4, dim_feedforward = 256, dropout_prob = 0.15)
main_rng, init_rng, dropout_init_rng = random.split(main_rng, 3)
params = encoder_block.init({'params' : init_rng, 'dropout' : dropout_init_rng}, x, True)['params']
# dropout is stochaastic hence we need to pass rng to the forward pass
main_rng, dropout_rng = random.split(main_rng)
out = encoder_block.apply({'params' : params}, x, train = True, rngs = {'dropout' : dropout_rng})
if out.shape == (4, 16, 128):
    print("encoder block test passed")
else :
    print("encoder block test not passed")



# testing encoder

main_rng, x_rng = random.split(main_rng)
x = random.normal(x_rng, (4, 16, 128))

transenc = TransformerEncoder(num_layers= 5, input_dim = 128,
                              num_heads = 4, dim_feedforward = 256,
                              dropout_prob = 0.15)

params = transenc.init({'params' : init_rng, 'dropout' : dropout_rng}, x, train = True)['params']
main_rng, dropout_apply_rng = random.split(main_rng)
transenc_bind = transenc.bind({'params' : params}, rngs = {'dropout' : dropout_apply_rng})
out = transenc_bind(x, train = True)
attn_maps = transenc_bind.get_attention_maps(x, train = True)
if out.shape == (4, 16, 128):
    print("Transformer Encoder test passed")
else :
    print("Transformer Encoder test not passed")
if attn_maps[0].shape == (4, 4, 16, 16):
    print("attention maps test passed")
else :
    print("attention maps test not passed")





