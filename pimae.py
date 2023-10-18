

from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from jax_mae import mae_vit_base_patch16 as mae
from utils_net_jax import FCN
from typing import Callable
import pdb


class low_rank_map(nn.Module):
    rank: int = 1
    @nn.compact
    def __call__(self, x):
        y = nn.Dense(features=x.shape[-1])(x)
        y = nn.gelu(y)
        y = nn.Dense(features=128)(y)
        y = nn.gelu(y)
        y = nn.Dense(features=128)(y)
        y = nn.gelu(y)
        y = nn.Dense(features=self.rank)(y)
        y = nn.Dense(features=128)(y)
        y = nn.gelu(y)
        y = nn.Dense(features=x.shape[-1])(y)
        return y
    
    
    
class Decoder(nn.Module):
    features: int = 64
    patch_size: tuple[int, int, int] = (1, 16, 16)
    out_chans: int = 1
    kernel_init: Callable = nn.initializers.kaiming_normal()
    bias_init: Callable = nn.initializers.zeros_init()
    
    @nn.compact
    def __call__(self, x):
        assert self.patch_size[1] == self.patch_size[2]
        up_scale_times = int(np.log2(self.patch_size[1]))
        f = jax.image.resize(x, shape=(x.shape[0], x.shape[1]*self.patch_size[0], x.shape[2], x.shape[3], x.shape[4]), method='nearest')
        for t in range(up_scale_times):
            f = jax.image.resize(f, shape=(f.shape[0], f.shape[1], f.shape[2]*2, f.shape[3]*2, f.shape[4]), method='nearest')
            features_num = 2 ** (up_scale_times - t - 1) * self.features
            f = nn.Conv(features_num, kernel_size=(1, 2, 2), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
            f = nn.relu(f)
            f = nn.Conv(features_num, kernel_size=(1, 3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
            f = nn.relu(f)
            f = nn.Conv(features_num, kernel_size=(1, 3, 3), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
            f = nn.relu(f)
        y = nn.Conv(self.out_chans, kernel_size=(1, 1, 1), kernel_init=self.kernel_init, bias_init=self.bias_init)(f)
        return y



class ViT_CNN(nn.Module):
    img_size: tuple[int, int, int] = (1, 192, 192)
    patch_size: tuple[int, int, int] = (1, 16, 16)
    out_chans: int = 1
    lorm_dim: int = None
    
    def setup(self):
        self.MAE = mae(img_size=self.img_size, PatchEmbed_type="mae3d", patch_size=self.patch_size)
        self.emitter_decoder = Decoder(patch_size=self.patch_size, out_chans=1)
        self.bg_decoder = Decoder(patch_size=self.patch_size, out_chans=1)
        self.lf_decoder = Decoder(patch_size=self.patch_size, out_chans=self.out_chans)
        if self.lorm_dim is not None:
            self.lorm = low_rank_map(self.lorm_dim)
        

    def unpatchify_feature(self, x):
        """
        x: (N, L, C)
        f: (N, D, H, W, C)
        """
        p = self.patch_size
        d, h, w = (s // p[i] for i, s in enumerate(self.img_size))
        f = x.reshape((x.shape[0], d, h, w, -1))
        return f

    def __call__(self, x, args, training, mask_ratio):
        # batch, C, Z, Y, X
        img_t = x.transpose([0, 2, 3, 4, 1])
        # ViT encoder
        rng = self.make_rng("random_masking")
        
        if training:
            mask_ratio = float(mask_ratio)
        else:
            mask_ratio = 0.0
        latent, mask, ids_restore = self.MAE.forward_encoder(img_t, mask_ratio=mask_ratio, train=training, rng=rng)
        laten_embed_to_blk = self.MAE.forward_decoder_embed(latent, ids_restore)
        Features = self.MAE.forward_decoder_blks(laten_embed_to_blk, train=training)[:, 1:, :]
        

        mask = jnp.tile(jnp.expand_dims(mask, -1), (1, 1, x.shape[1]*self.patch_size[0]*self.patch_size[1]*self.patch_size[2]))
        mask = self.MAE.unpatchify(mask)
        mask = mask.transpose([0, 4, 1, 2, 3])

        f = self.unpatchify_feature(Features)
        # emitter
        emitter = self.emitter_decoder(f).transpose([0, 4, 1, 2, 3])
        # background
        bg = nn.gelu(self.bg_decoder(f))
        bg = nn.avg_pool(bg, tuple([self.patch_size[0], self.patch_size[1]*4, self.patch_size[2]*4]), padding="SAME")
        bg = bg.transpose([0, 4, 1, 2, 3])

        if args.lorm_dim is not None:
            f = self.lorm(f)
        
        lightfield = self.lf_decoder(f).transpose([0, 4, 1, 2, 3])

        return emitter, bg, lightfield, mask



class PiMAE(nn.Module):
    img_size: tuple[int, int, int] = (224, 224, 128)
    patch_size: tuple[int, int, int] = (16, 16, 16)
    psf_size: tuple[int, int, int] = (64, 64, 64)
    out_chans: int = 1
    lorm_dim: int = 1


    def setup(self):
        # emitter
        self.pt_predictor = ViT_CNN(self.img_size, self.patch_size, self.out_chans, self.lorm_dim)
        # PSF
        self.psf_seed = self.param("psf_seed", nn.initializers.normal(1), [1, 32, *self.psf_size])
        self.PSF_predictor = FCN(1)

        
    def __call__(self, x_clean, args, training):
        # rng = self.make_rng("random_masking")
        rng = jax.random.PRNGKey(0)
        rng_noise_1, rng_noise_2, rng_noise_3 = jax.random.split(rng, 3)

        # x : [batch, channel, z, y, x]
        x = x_clean
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1], x.shape[2]*args.rescale[0], x.shape[3]*args.rescale[1], x.shape[4]*args.rescale[2]), method='linear')
        if training:
            x_mean = jnp.mean(x, axis=[1, 2, 3, 4], keepdims=True)
            noise = x_mean * jax.lax.stop_gradient(jax.random.normal(rng_noise_1, x.shape) \
                            * jax.random.uniform(rng_noise_2, (x.shape[0], 1, 1, 1, 1), minval=0.0, maxval=args.add_noise))
            x = x + noise
        
        emitter, bg, lightfield, mask = self.pt_predictor(x, args, training, args.mask_ratio)
        if args.resume_psf_path is not None:
            emitter, bg = nn.relu(emitter), nn.relu(bg)
        else:
            emitter, bg = nn.gelu(emitter), nn.gelu(bg)
        
        lightfield = jax.nn.softmax(lightfield, axis=1) * lightfield.shape[1]

        S = lightfield * emitter
        
        # PSF
        if args.resume_psf_path is not None:
            psf = jax.lax.stop_gradient(self.PSF_predictor(jax.lax.stop_gradient(self.psf_seed), training))
        else:
            psf = self.PSF_predictor(jax.lax.stop_gradient(self.psf_seed), training)
        psf = jax.nn.softmax(psf, axis=(2, 3, 4))
        # rec
        rec = convolve(S, psf) + bg
        # rec_real = jax.image.resize(rec, shape=(x.shape[0], x.shape[1], x.shape[2]//args.rescale[0], x.shape[3]//args.rescale[1], x.shape[4]//args.rescale[2]), method='linear')
        rec_real = nn.avg_pool(rec.transpose([0, 2, 3, 4, 1]), tuple(args.rescale), tuple(args.rescale), padding="VALID").transpose([0, 4, 1, 2, 3])
        mask_real = nn.avg_pool(mask.transpose([0, 2, 3, 4, 1]), tuple(args.rescale), tuple(args.rescale), padding="VALID").transpose([0, 4, 1, 2, 3])
        return {
            "x_real": x_clean,
            "x_up": x*(1-mask), 
            "deconv": emitter,
            "lightfield": lightfield,
            "background": bg,
            "rec_real": rec_real,
            "rec_up": rec,
            "psf": psf,
            "mask": mask_real
        }



def convolve(xin, k):
    if xin.shape[2] > 1:
        x = xin.reshape([-1, 1, *xin.shape[2:]])
        dn = jax.lax.conv_dimension_numbers(x.shape, k.shape,('NCHWD', 'IOHWD', 'NCHWD'))
        y = jax.lax.conv_general_dilated(x, k, window_strides =(1, 1, 1), dimension_numbers=dn, padding='SAME', precision='highest')
        return y.reshape([*xin.shape[:2], *y.shape[-3:]])
    else:
        x = xin.reshape([-1, 1, *xin.shape[3:]])
        k = k.reshape([1, 1, k.shape[3], k.shape[4]])
        dn = jax.lax.conv_dimension_numbers(x.shape, k.shape,('NCHW', 'IOHW', 'NCHW'))
        y = jax.lax.conv_general_dilated(x, k, window_strides =(1, 1), dimension_numbers=dn, padding='SAME', precision='highest')
        return y.reshape([*xin.shape[:2], 1, *y.shape[-2:]])


def instance_normalize(x):
    x_min = jnp.min(x, axis=[1, 2, 3, 4], keepdims=True)
    x = x - x_min
    x_mean = jnp.mean(x, axis=[1, 2, 3, 4], keepdims=True)
    x_mean = jnp.maximum(x_mean, 1e-3)
    x = x / x_mean
    x = jax.lax.stop_gradient(x)
    x_min = jax.lax.stop_gradient(x_min)
    x_mean = jax.lax.stop_gradient(x_mean)
    return x, x_min, x_mean



import argparse
import optax
parser = argparse.ArgumentParser(description='Super-resolved microscopy via physics-informed masked autoencoder')
##################################### Dataset #################################################
parser.add_argument('--crop_size', nargs='+', type=int, default=[1, 14, 14])
parser.add_argument('--trainset', type=str, default="../data/3D")
parser.add_argument('--testset', type=str, default="../data/3D")
parser.add_argument('--eval_dir', type=str, default=None)
parser.add_argument('--min_datasize', type=int, default=16000)
##################################### Training setting #################################################
parser.add_argument('--batchsize', type=int, default=16)
parser.add_argument('--resume_path', type=str, default=None)
parser.add_argument('--resume_pretrain_path', type=str, default=None)
parser.add_argument('--resume_psf_path', type=str, default=None)
parser.add_argument('--epoch', type=int, default=81)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--snapshot_epoch', type=int, default=1)
parser.add_argument('--eval_epoch', type=int, default=1)
parser.add_argument('--eval_image_num', type=int, default=5)
parser.add_argument('--eval_sigma', type=int, default=None)
parser.add_argument('--add_noise', type=float, default=1)
parser.add_argument('--decay_steps_ratio', type=float, default=0.5)
##################################### Loss ############################################
parser.add_argument('--tv_loss', type=float, default=1e-3)
parser.add_argument('--lf_tv', type=float, default=0)
parser.add_argument('--lf_l1', type=float, default=0)
##################################### MAE ############################################
parser.add_argument('--mask_ratio', type=float, default=0.75)
parser.add_argument('--patch_size', nargs='+', type=int, default=[1, 16, 16])
##################################### Physics ############################################
parser.add_argument('--psf_size', nargs='+', type=int, default=[1, 33, 33])
parser.add_argument('--psfc_loss', type=float, default=1e-1)
parser.add_argument('--rescale', nargs='+', type=int, default=[1, 1, 1])
##################################### net ############################################
parser.add_argument('--lorm_dim', type=int, default=None)
##################################### Log ############################################
parser.add_argument('--save_dir', type=str, default="./ckpt")
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()

from typing import Any
from jax import lax
import jax.numpy as jnp
from flax.training import train_state

class TrainState(train_state.TrainState):
  batch_stats: Any
def create_train_state(rng):
    net = PiMAE()
    x_init = jax.random.normal(rng, (1, 9, *args.crop_size))
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    variables = net.init({"params": rng1, 'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, x_init, args, True)
    return TrainState.create(
        apply_fn=net.apply, params=variables['params'], batch_stats=variables['batch_stats'], tx=optax.adamw(0.0))

rng = jax.random.PRNGKey(42)
rng, init_rng = jax.random.split(rng)
state = create_train_state(init_rng)
    



if __name__ == '__main__':
    # input [batch, channel, 1, h, w]
    input = jax.numpy.ones((2,9,*args.crop_size))
    x = jax.numpy.zeros((2,9,*args.crop_size))
    print(x.shape)
    rng = jax.random.PRNGKey(0)
    rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
    x_init = input
    # variables = net.init({"params": rng1, 'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, x_init, args, True)
    result, updates = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, x, args, True, rngs={'dropout': rng2, "random_masking": rng3, "drop_path": rng4}, mutable=['batch_stats'])
    import pdb
    pdb.set_trace()
    print('printing output')

