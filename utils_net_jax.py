
from flax import linen as nn
import jax
import jax.numpy as jnp
# from utils_flax import MaskedConv
# from typing import Callable


# class Encoder(nn.Module):
#     features: int = 64
#     kernel_init: Callable = nn.initializers.kaiming_normal()
#     bias_init: Callable = nn.initializers.zeros_init()

#     @nn.compact
#     def __call__(self, x, training):
#         z1 = nn.Conv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(x)
#         z1 = nn.BatchNorm(use_running_average=False)(z1)
#         z1 = nn.relu(z1)
#         z1 = nn.Conv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z1)
#         z1 = nn.BatchNorm(use_running_average=False)(z1)
#         z1 = nn.relu(z1)
#         z1_pool = nn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))

#         z2 = nn.Conv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z1_pool)
#         z2 = nn.BatchNorm(use_running_average=False)(z2)
#         z2 = nn.relu(z2)
#         z2 = nn.Conv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z2)
#         z2 = nn.BatchNorm(use_running_average=False)(z2)
#         z2 = nn.relu(z2)
#         z2_pool = nn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))

#         z3 = nn.Conv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z2_pool)
#         z3 = nn.BatchNorm(use_running_average=False)(z3)
#         z3 = nn.relu(z3)
#         z3 = nn.Conv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z3)
#         z3 = nn.BatchNorm(use_running_average=False)(z3)
#         z3 = nn.relu(z3)
#         z3_pool = nn.max_pool(z3, window_shape=(2, 2), strides=(2, 2))

#         z4 = nn.Conv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z3_pool)
#         z4 = nn.BatchNorm(use_running_average=False)(z4)
#         z4 = nn.relu(z4)
#         z4 = nn.Conv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z4)
#         z4 = nn.BatchNorm(use_running_average=False)(z4)
#         z4 = nn.relu(z4)
#         z4_dropout = nn.Dropout(0.5, deterministic=not training)(z4)
#         z4_pool = nn.max_pool(z4_dropout, window_shape=(2, 2), strides=(2, 2))

#         z5 = nn.Conv(self.features * 16, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z4_pool)
#         z5 = nn.BatchNorm(use_running_average=False)(z5)
#         z5 = nn.relu(z5)
#         z5 = nn.Conv(self.features * 16, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z5)
#         z5 = nn.BatchNorm(use_running_average=False)(z5)
#         z5 = nn.relu(z5)
#         z5_dropout = nn.Dropout(0.5, deterministic=not training)(z5)

#         return z1, z2, z3, z4_dropout, z5_dropout


# class Decoder(nn.Module):
#     features: int = 64
#     out: int = 1
#     kernel_init: Callable = nn.initializers.kaiming_normal()
#     bias_init: Callable = nn.initializers.zeros_init()

#     @nn.compact
#     def __call__(self, z1, z2, z3, z4_dropout, z5_dropout, training):
#         z6_up = jax.image.resize(z5_dropout, shape=(z5_dropout.shape[0], z5_dropout.shape[1] * 2, z5_dropout.shape[2] * 2, z5_dropout.shape[3]),
#                                  method='nearest')
#         z6 = nn.Conv(self.features * 8, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z6_up)
#         z6 = nn.BatchNorm(use_running_average=False)(z6)
#         z6 = nn.relu(z6)
#         z6 = jnp.concatenate([z4_dropout, z6], axis=3)
#         z6 = nn.Conv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z6)
#         z6 = nn.BatchNorm(use_running_average=False)(z6)
#         z6 = nn.relu(z6)
#         z6 = nn.Conv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z6)
#         z6 = nn.BatchNorm(use_running_average=False)(z6)
#         z6 = nn.relu(z6)

#         z7_up = jax.image.resize(z6, shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
#                                  method='nearest')
#         z7 = nn.Conv(self.features * 4, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z7_up)
#         z7 = nn.BatchNorm(use_running_average=False)(z7)
#         z7 = nn.relu(z7)
#         z7 = jnp.concatenate([z3, z7], axis=3)
#         z7 = nn.Conv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z7)
#         z7 = nn.BatchNorm(use_running_average=False)(z7)
#         z7 = nn.relu(z7)
#         z7 = nn.Conv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z7)
#         z7 = nn.BatchNorm(use_running_average=False)(z7)
#         z7 = nn.relu(z7)

#         z8_up = jax.image.resize(z7, shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2] * 2, z7.shape[3]),
#                                  method='nearest')
#         z8 = nn.Conv(self.features * 2, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z8_up)
#         z8 = nn.BatchNorm(use_running_average=False)(z8)
#         z8 = nn.relu(z8)
#         z8 = jnp.concatenate([z2, z8], axis=3)
#         z8 = nn.Conv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z8)
#         z8 = nn.BatchNorm(use_running_average=False)(z8)
#         z8 = nn.relu(z8)
#         z8 = nn.Conv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z8)
#         z8 = nn.BatchNorm(use_running_average=False)(z8)
#         z8 = nn.relu(z8)

#         z9_up = jax.image.resize(z8, shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2] * 2, z8.shape[3]),
#                                  method='nearest')
#         z9 = nn.Conv(self.features, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9_up)
#         z9 = nn.relu(z9)
#         z9 = jnp.concatenate([z1, z9], axis=3)
#         z9 = nn.Conv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9)
#         z9 = nn.relu(z9)
#         z9 = nn.Conv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9)
#         z9 = nn.relu(z9)
#         y = nn.Conv(self.out, kernel_size=(1, 1), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9)
#         # y = nn.Conv(self.out, kernel_size=(1, 1), kernel_init=nn.initializers.zeros, bias_init=self.bias_init)(z9)
#         return y


# class UNet(nn.Module):
#     features: int = 64
#     out: int = 1

#     @nn.compact
#     def __call__(self, x, training):
#         x = jnp.transpose(x, (0, 2, 3, 1))
#         z1, z2, z3, z4_dropout, z5_dropout = Encoder(self.features)(x, training)
#         y = Decoder(self.features, out=self.out)(z1, z2, z3, z4_dropout, z5_dropout, training)
#         y = jnp.transpose(y, (0, 3, 1, 2))
#         return y


class FCN(nn.Module):
    out: int = 1
    @nn.compact
    def __call__(self, x, training):
        x = jnp.transpose(x, (0, 2, 3, 4, 1))
        x = nn.Conv(features=48, kernel_size=[3, 3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(features=96, kernel_size=[3, 3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(features=96, kernel_size=[3, 3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x)
        x = nn.BatchNorm(use_running_average=False)(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(features=48, kernel_size=[3, 3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x)
        x = nn.activation.leaky_relu(x)
        x = nn.Conv(features=self.out, kernel_size=[3, 3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x)
        x = jnp.transpose(x, (0, 4, 1, 2, 3))
        return x

# class MaskedFCN(nn.Module):
#     out: int = 1
#     dropconnect_rate: float = 0.5
    
#     @nn.compact
#     def __call__(self, x, training):
#         x = jnp.transpose(x, (0, 2, 3, 1))
#         x = MaskedConv(features=48, kernel_size=[3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x, self.dropconnect_rate)
#         x = nn.BatchNorm(use_running_average=False)(x)
        
#         x = nn.activation.leaky_relu(x)
#         x = MaskedConv(features=96, kernel_size=[3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x, self.dropconnect_rate)
#         x = nn.BatchNorm(use_running_average=False)(x)
       
#         x = nn.activation.leaky_relu(x)
#         x = MaskedConv(features=96, kernel_size=[3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x, self.dropconnect_rate)
#         x = nn.BatchNorm(use_running_average=False)(x)
        
#         x = nn.activation.leaky_relu(x)
#         x = MaskedConv(features=48, kernel_size=[3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x, self.dropconnect_rate)
#         x = nn.activation.leaky_relu(x)
        
#         x = nn.Conv(features=self.out, kernel_size=[3, 3], kernel_init=nn.initializers.kaiming_normal(), bias_init=nn.initializers.zeros_init(), padding='same')(x)
#         x = jnp.transpose(x, (0, 3, 1, 2))
#         return x

# class MaskedEncoder(nn.Module):
#     features: int = 64
#     kernel_init: Callable = nn.initializers.lecun_normal()
#     bias_init: Callable = nn.initializers.zeros
#     drop_rate: float = 0
#     dropconnect_rate: float = 0

#     @nn.compact
#     def __call__(self, x, training):
#         z1 = MaskedConv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(x, self.dropconnect_rate)
#         z1 = nn.BatchNorm(use_running_average=False)(z1)
#         z1 = nn.relu(z1)
#         z1 = MaskedConv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z1, self.dropconnect_rate)
#         z1 = nn.BatchNorm(use_running_average=False)(z1)
#         z1 = nn.relu(z1)
#         z1_pool = nn.max_pool(z1, window_shape=(2, 2), strides=(2, 2))

#         z2 = MaskedConv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z1_pool, self.dropconnect_rate)
#         z2 = nn.BatchNorm(use_running_average=False)(z2)
#         z2 = nn.relu(z2)
#         z2 = MaskedConv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z2, self.dropconnect_rate)
#         z2 = nn.BatchNorm(use_running_average=False)(z2)
#         z2 = nn.relu(z2)
#         z2_pool = nn.max_pool(z2, window_shape=(2, 2), strides=(2, 2))

#         z3 = MaskedConv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z2_pool, self.dropconnect_rate)
#         z3 = nn.BatchNorm(use_running_average=False)(z3)
#         z3 = nn.relu(z3)
#         z3 = MaskedConv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z3, self.dropconnect_rate)
#         z3 = nn.BatchNorm(use_running_average=False)(z3)
#         z3 = nn.relu(z3)
#         z3 = nn.Dropout(self.drop_rate, deterministic=not training)(z3)
#         z3_pool = nn.max_pool(z3, window_shape=(2, 2), strides=(2, 2))

#         z4 = MaskedConv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z3_pool, self.dropconnect_rate)
#         z4 = nn.BatchNorm(use_running_average=False)(z4)
#         z4 = nn.relu(z4)
#         z4 = MaskedConv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z4, self.dropconnect_rate)
#         z4 = nn.BatchNorm(use_running_average=False)(z4)
#         z4 = nn.relu(z4)
#         z4_dropout = nn.Dropout(self.drop_rate, deterministic=not training)(z4)
#         z4_pool = nn.max_pool(z4_dropout, window_shape=(2, 2), strides=(2, 2))

#         z5 = MaskedConv(self.features * 16, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z4_pool, self.dropconnect_rate)
#         z5 = nn.BatchNorm(use_running_average=False)(z5)
#         z5 = nn.relu(z5)
#         z5 = MaskedConv(self.features * 16, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z5, self.dropconnect_rate)
#         z5 = nn.BatchNorm(use_running_average=False)(z5)
#         z5 = nn.relu(z5)
#         z5_dropout = nn.Dropout(self.drop_rate, deterministic=not training)(z5)

#         return z1, z2, z3, z4_dropout, z5_dropout


# class MaskedDecoder(nn.Module):
#     features: int = 64
#     out: int = 1
#     kernel_init: Callable = nn.initializers.lecun_normal()
#     bias_init: Callable = nn.initializers.zeros_init()
#     dropconnect_rate: float = 0

#     @nn.compact
#     def __call__(self, z1, z2, z3, z4_dropout, z5_dropout, training):
#         z6_up = jax.image.resize(z5_dropout, shape=(z5_dropout.shape[0], z5_dropout.shape[1] * 2, z5_dropout.shape[2] * 2, z5_dropout.shape[3]),
#                                  method='nearest')
#         z6 = MaskedConv(self.features * 8, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z6_up, self.dropconnect_rate)
#         z6 = nn.BatchNorm(use_running_average=False)(z6)
#         z6 = nn.relu(z6)
#         z6 = jnp.concatenate([z4_dropout, z6], axis=3)
#         z6 = MaskedConv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z6, self.dropconnect_rate)
#         z6 = nn.BatchNorm(use_running_average=False)(z6)
#         z6 = nn.relu(z6)
#         z6 = MaskedConv(self.features * 8, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z6, self.dropconnect_rate)
#         z6 = nn.BatchNorm(use_running_average=False)(z6)
#         z6 = nn.relu(z6)

#         z7_up = jax.image.resize(z6, shape=(z6.shape[0], z6.shape[1] * 2, z6.shape[2] * 2, z6.shape[3]),
#                                  method='nearest')
#         z7 = MaskedConv(self.features * 4, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z7_up, self.dropconnect_rate)
#         z7 = nn.BatchNorm(use_running_average=False)(z7)
#         z7 = nn.relu(z7)
#         z7 = jnp.concatenate([z3, z7], axis=3)
#         z7 = MaskedConv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z7, self.dropconnect_rate)
#         z7 = nn.BatchNorm(use_running_average=False)(z7)
#         z7 = nn.relu(z7)
#         z7 = MaskedConv(self.features * 4, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z7, self.dropconnect_rate)
#         z7 = nn.BatchNorm(use_running_average=False)(z7)
#         z7 = nn.relu(z7)

#         z8_up = jax.image.resize(z7, shape=(z7.shape[0], z7.shape[1] * 2, z7.shape[2] * 2, z7.shape[3]),
#                                  method='nearest')
#         z8 = MaskedConv(self.features * 2, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z8_up, self.dropconnect_rate)
#         z8 = nn.BatchNorm(use_running_average=False)(z8)
#         z8 = nn.relu(z8)
#         z8 = jnp.concatenate([z2, z8], axis=3)
#         z8 = MaskedConv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z8, self.dropconnect_rate)
#         z8 = nn.BatchNorm(use_running_average=False)(z8)
#         z8 = nn.relu(z8)
#         z8 = MaskedConv(self.features * 2, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z8, self.dropconnect_rate)
#         z8 = nn.BatchNorm(use_running_average=False)(z8)
#         z8 = nn.relu(z8)

#         z9_up = jax.image.resize(z8, shape=(z8.shape[0], z8.shape[1] * 2, z8.shape[2] * 2, z8.shape[3]),
#                                  method='nearest')
#         z9 = MaskedConv(self.features, kernel_size=(2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9_up, self.dropconnect_rate)
#         z9 = nn.BatchNorm(use_running_average=False)(z9)
#         z9 = nn.relu(z9)
#         z9 = jnp.concatenate([z1, z9], axis=3)
#         z9 = MaskedConv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9, self.dropconnect_rate)
#         z9 = nn.BatchNorm(use_running_average=False)(z9)
#         z9 = nn.relu(z9)
#         z9 = MaskedConv(self.features, kernel_size=(3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9, self.dropconnect_rate)
#         z9 = nn.BatchNorm(use_running_average=False)(z9)
#         y = nn.Conv(self.out, kernel_size=(1, 1), kernel_init=self.kernel_init, bias_init=self.bias_init)(z9)
#         return y


# class MaskedUNet(nn.Module):
#     features: int = 64
#     out: int = 1
#     drop_rate: float = 0
#     dropconnect_rate: float = 0

#     @nn.compact
#     def __call__(self, xin, training):
#         z1_list = []
#         z2_list = []
#         z3_list = []
#         z4_dropout_list = []
#         z5_dropout_list = []
        
#         for c in range(xin.shape[1]):
#             x = xin[:, c:c+1, :, :]
#             x = jnp.transpose(x, (0, 2, 3, 1))
#             z1, z2, z3, z4_dropout, z5_dropout = MaskedEncoder(self.features, drop_rate=self.drop_rate, dropconnect_rate=self.dropconnect_rate)(x, training)
#             z1_list.append(z1)
#             z2_list.append(z2)
#             z3_list.append(z3)
#             z4_dropout_list.append(z4_dropout)
#             z5_dropout_list.append(z5_dropout)
        
#         y = MaskedDecoder(self.features, out=self.out, dropconnect_rate=self.dropconnect_rate)(sum(z1_list), sum(z2_list), sum(z3_list), sum(z4_dropout_list), sum(z5_dropout_list), training)
#         y = jnp.transpose(y, (0, 3, 1, 2))
#         return y

def relu_mask(x, threshold=0.):
    return jax.nn.relu(x - threshold) + threshold


