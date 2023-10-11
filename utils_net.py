import torch
import torch.nn as nn
from math import floor
import pdb
_drop = 0 #与network中的_drop保持一致

class Crop3d(nn.Module):

    def __init__(self, crop):
        super().__init__()
        self.crop = crop
        assert len(crop) == 4

    def forward(self, x):
        (X_s, X_e, Y_s, Y_e) = self.crop
        x0, x1 = X_s, x.shape[-1] - X_e
        y0, y1 = Y_s, x.shape[-2] - Y_e
        return x[:, :, y0:y1, x0:x1] #BCYX


class ZeroPad3d(nn.Module):

    def __init__(self, pad):
        super().__init__()
        self.pad = pad
        assert len(pad) == 4
    
    def forward(self, x):
        (x_a, x_b, y_a, y_b) = self.pad
        return nn.functional.pad(x, (x_a, x_b, y_a, y_b))


class Shift3d(nn.Module):

    def __init__(self, shift):
        super().__init__()
        self.shift = shift
        x_shift, y_shift = self.shift
        y_a, y_b = abs(y_shift), 0
        x_a, x_b = abs(x_shift), 0
        if y_shift < 0:
            y_a, y_b = y_b, y_a
        if x_shift < 0:
            x_a, x_b = x_b, x_a
        # Order : x, y, z
        self.pad = ZeroPad3d((x_a, x_b, y_a, y_b))
        self.crop = Crop3d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x):
        return self.shift_block(x)

class BayesianShiftConv3d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        """
        https://arxiv.org/pdf/1506.02142v6.pdf 
        Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning
        """
        super().__init__(*args, **kwargs)
        self.shift_size = (self.kernel_size[0] // 2, 0)
        # Use individual layers of shift for wrapping conv with shift
        shift = Shift3d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x):
        x = self.pad(x)
        x = super().forward(x)
        x = self.crop(x)
        return x



def rotate_3d(x, side):
    """Rotate images by Six sides. Can handle any 3D (BCZYX) BCHW data format.
    Args:
        x (Tensor): Image or batch of images.
        side (int): 0:x+, 1:x-, 2:y+, 3:y-
    Returns:
        Tensor: Copy of tensor with rotation applied.
    """

    if side == 0:
        return x
    elif side == 1:
        return x.flip(-1).flip(-2)
    elif side == 2:
        return x.flip(-1).transpose(-1, -2)
    elif side == 3:
        return x.flip(-2).transpose(-1, -2)
    else:
        raise NotImplementedError("Must be one of 0-7 sides")


def rotate_3d_re(x, side):
    """Rotate images by Six sides. Can handle any 4D (BCTZYX) BCHW data format.
    Args:
        x (Tensor): Image or batch of images.
        side (int): 0:x+, 1:x-, 2:y+, 3:y-
    Returns:
        Tensor: Copy of tensor with rotation applied.
    """

    if side == 0:
        return x
    elif side == 1:
        return x.flip(-2).flip(-1)
    elif side == 2:
        return x.transpose(-1, -2).flip(-1)
    elif side == 3:
        return x.transpose(-1, -2).flip(-2)
    else:
        raise NotImplementedError("Must be one of 0-7 sides")