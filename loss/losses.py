import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg_arch import VGGFeatureExtractor, Registry
from loss.loss_utils import *


_reduction_modes = ['none', 'mean', 'sum']

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
        
        
        
class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda()

        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight


class ChromaLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(ChromaLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred_hvi, target_hvi, **kwargs):
        pred_hv = pred_hvi[:, 0:2, :, :]
        target_hv = target_hvi[:, 0:2, :, :]
        return self.loss_weight * l1_loss(pred_hv, target_hv, reduction=self.reduction)


class DarkChromaLoss(nn.Module):
    def __init__(self, loss_weight=1.0, dark_threshold=0.4, reduction='mean', min_mask=1e-6):
        super(DarkChromaLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.dark_threshold = float(dark_threshold)
        self.reduction = reduction
        self.min_mask = float(min_mask)

    def forward(self, pred_hvi, target_hvi, **kwargs):
        pred_hv = pred_hvi[:, 0:2, :, :]
        target_hv = target_hvi[:, 0:2, :, :]
        target_i = target_hvi[:, 2:3, :, :]
        mask = (target_i < self.dark_threshold).float()
        if mask.sum().item() <= self.min_mask:
            return pred_hv.new_tensor(0.0)
        return self.loss_weight * l1_loss(pred_hv, target_hv, weight=mask, reduction=self.reduction)

class ColorMapLoss(nn.Module):
    """
    局部颜色均值损失。
    作用：让输出图在局部区域的 RGB 色彩分布接近 GT。
    比单纯 MeanColorLoss 更适合修复“植物不绿、墙面偏色、大块色斑”。
    """
    def __init__(self, loss_weight=1.0, kernel_size=32, stride=16):
        super(ColorMapLoss, self).__init__()
        self.loss_weight = float(loss_weight)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)

    def forward(self, pred, target):
        # 全局颜色均值
        global_loss = F.l1_loss(
            pred.mean(dim=(2, 3)),
            target.mean(dim=(2, 3))
        )

        h, w = pred.shape[-2], pred.shape[-1]

        # 局部颜色均值，输入 patch 足够大时启用
        if h >= self.kernel_size and w >= self.kernel_size:
            pred_map = F.avg_pool2d(
                pred,
                kernel_size=self.kernel_size,
                stride=self.stride
            )
            target_map = F.avg_pool2d(
                target,
                kernel_size=self.kernel_size,
                stride=self.stride
            )
            local_loss = F.l1_loss(pred_map, target_map)
        else:
            local_loss = pred.new_tensor(0.0)

        return self.loss_weight * (global_loss + local_loss)

class DarkHVSmoothLoss(nn.Module):
    """
    暗区 H/V 平滑损失。
    作用：抑制暗部和平坦区域的彩色噪声、绿色/红色块状污染。
    """
    def __init__(self, loss_weight=0.05, dark_threshold=0.45):
        super(DarkHVSmoothLoss, self).__init__()
        self.loss_weight = float(loss_weight)
        self.dark_threshold = float(dark_threshold)

    def forward(self, pred_hvi, target_hvi):
        pred_hv = pred_hvi[:, 0:2, :, :]
        target_i = target_hvi[:, 2:3, :, :]

        mask = (target_i < self.dark_threshold).float()

        dx = torch.abs(pred_hv[:, :, :, 1:] - pred_hv[:, :, :, :-1])
        dy = torch.abs(pred_hv[:, :, 1:, :] - pred_hv[:, :, :-1, :])

        mask_x = mask[:, :, :, 1:]
        mask_y = mask[:, :, 1:, :]

        loss_x = (dx * mask_x).mean()
        loss_y = (dy * mask_y).mean()

        return self.loss_weight * (loss_x + loss_y)

class FlatRGBSmoothLoss(nn.Module):
    """
    平坦区域 RGB 平滑损失。
    用 GT 的梯度判断哪些地方是平坦区，只在平坦区抑制输出图的高频噪声。
    用于减少墙面、地面、椅子上的颗粒噪声。
    """
    def __init__(self, loss_weight=0.05, edge_threshold=0.03):
        super(FlatRGBSmoothLoss, self).__init__()
        self.loss_weight = float(loss_weight)
        self.edge_threshold = float(edge_threshold)

    def rgb_to_gray(self, x):
        return 0.299 * x[:, 0:1, :, :] + 0.587 * x[:, 1:2, :, :] + 0.114 * x[:, 2:3, :, :]

    def forward(self, pred, target):
        target_gray = self.rgb_to_gray(target)

        target_dx = torch.abs(target_gray[:, :, :, 1:] - target_gray[:, :, :, :-1])
        target_dy = torch.abs(target_gray[:, :, 1:, :] - target_gray[:, :, :-1, :])

        flat_mask_x = (target_dx < self.edge_threshold).float()
        flat_mask_y = (target_dy < self.edge_threshold).float()

        pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])

        loss_x = (pred_dx * flat_mask_x).mean()
        loss_y = (pred_dy * flat_mask_y).mean()

        return self.loss_weight * (loss_x + loss_y)

class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss




class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True,weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight
