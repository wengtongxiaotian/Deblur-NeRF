import torch
import torch.nn as nn
from utils_net import rotate_3d, rotate_3d_re, Shift3d, BayesianShiftConv3d
import pdb

Conv3d = BayesianShiftConv3d

class UNet_3d(nn.Module):
    def __init__(self, in_channels=1):
        super(UNet_3d, self).__init__()
        
        ####################################
        # Encode Blocks
        ####################################
        def _max_pool_block(max_pool):
            return nn.Sequential(Shift3d((1, 0, 0)), max_pool)

        # Layers: enc_conv0, enc_conv1, pool1
        self.encode_block_1 = nn.Sequential(
            Conv3d(in_channels, 48, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Conv3d(48, 48, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Separate instances of same encode module definition created
        self.encode_block_2 = nn.Sequential(
            Conv3d(48, 48, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        def _encode_block_3_4_5():
            return nn.Sequential(
                Conv3d(48, 48, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                _max_pool_block(nn.AvgPool3d(2)),
        )

        self.encode_block_3 = _encode_block_3_4_5()
        self.encode_block_4 = _encode_block_3_4_5()
        self.encode_block_5 = _encode_block_3_4_5()
        self.encode_block_6 = nn.Sequential(
                Conv3d(48, 48, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Decode Blocks
        ####################################
        # Layers: upsample5
        self.decode_block_6 = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"))

        # Layers: dec_conv5a, dec_conv5b, upsample4
        self.decode_block_5 = nn.Sequential(
                Conv3d(96, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                Conv3d(96, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
        )
        self.decode_block_4 = nn.Sequential(
                Conv3d(144, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                Conv3d(96, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Upsample(scale_factor=2, mode="nearest"),
        )

        # Layers: dec_deconv(i)a, dec_deconv(i)b, upsample(i-1); i=4..2
        def _decode_block_3_2():
            return nn.Sequential(
                Conv3d(144, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                Conv3d(96, 96, 3, stride=1, padding=1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        # Separate instances of same decode module definition created
        self.decode_block_3 = _decode_block_3_2()
        self.decode_block_2 = _decode_block_3_2()

        # Layers: dec_conv1a, dec_conv1b, dec_conv1c,
        self.decode_block_1 = nn.Sequential(
            Conv3d(96 + in_channels, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Conv3d(96, 96, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        ####################################
        # Output Block
        ####################################
        self.shift = Shift3d((1, 0, 0))
        nin_a_io = 96 * 8

        # nin_a,b,c, linear_act
        self.output_conv = Conv3d(96, 1, 1)
        self.output_block = nn.Sequential(
            Conv3d(nin_a_io, nin_a_io, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Conv3d(nin_a_io, 96, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            self.output_conv,
        )

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initializes weights using Kaiming  He et al. (2015).

        Only convolution layers have learnable weights. All convolutions use a leaky
        relu activation function (negative_slope = 0.1) except the last which is just
        a linear output.
        """
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, Conv3d):
                    nn.init.kaiming_normal_(m.weight.data, a=0.1)
                    if m.bias is not None:
                        m.bias.data.zero_()
            nn.init.kaiming_normal_(self.output_conv.weight.data)

    def forward(self, x, _drop=0):
        rotated = [rotate_3d(x, rot) for rot in range(8)]
        x = torch.cat((rotated), dim=0)
        # Encoder
        pool1 = self.encode_block_1(x) # 1
        pool2 = self.encode_block_2(pool1) # 1
        pool3 = self.encode_block_3(pool2) # 1/2
        pool4 = self.encode_block_4(pool3) # 1/4
        pool5 = self.encode_block_5(pool4) # 1/8
        encoded = self.encode_block_6(pool5) # 1/8

        # Decoder
        upsample5 = self.decode_block_6(encoded) # 1/4
        concat5 = torch.cat((upsample5, pool4), dim=1)
        upsample4 = self.decode_block_5(concat5)  # 1/2 96
        concat4 = torch.cat((upsample4, pool3), dim=1) # 144
        upsample3 = self.decode_block_4(concat4) # 1
        concat3 = torch.cat((upsample3, pool2), dim=1) # 144
        upsample2 = self.decode_block_3(concat3)
        concat2 = torch.cat((upsample2, pool1), dim=1) # 144
        upsample1 = self.decode_block_2(concat2)
        concat1 = torch.cat((upsample1, x), dim=1)
        x = self.decode_block_1(concat1)

        shifted = self.shift(x)
        # Unstack, rotate and combine
        rotated_batch = torch.chunk(shifted, 8, dim=0)
        aligned = [
            rotate_3d_re(rotated, rot)
            for rotated, rot in zip(rotated_batch, range(8))
        ]
        x = torch.cat(aligned, dim=1)
        x = self.output_block(x)
        return x
if __name__ == '__main__':
    unet = UNet_3d(in_channels=7)
    x = torch.zeros(2,7,100,100,100)
    y = unet(x)
    print('end')



